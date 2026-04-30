import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from scipy.optimize import minimize, Bounds, LinearConstraint
import numpy as np
import math
class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')
        
        # Subscriptions & Publishers
        self.create_subscription(PoseStamped, '/waypoints', self.waypoint_callback, 10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # --- CONFIGURATION ---
        self.horizon = 5         # N steps
        self.dt = 0.2           # Time step (s)
        
        # Limits (Hard Constraints)
        self.max_v = 0.22       # m/s
        self.max_w = 2.0        # rad/s
        self.max_acc_v = 0.2    # m/s^2 (Max linear acceleration)
        self.max_acc_w = 2.0    # rad/s^2 (Max angular acceleration)
        
        # Actuator Dynamics (First Order Lag)
        # v_actual(k+1) = v_actual(k) + alpha * (v_cmd - v_actual(k)) * dt
        self.tau_v = 0.5        # Time constant for linear velocity
        self.tau_w = 0.2        # Time constant for angular velocity
        self.alpha_v = 1.0 / self.tau_v
        self.alpha_w = 1.0 / self.tau_w
        # State: [x, y, theta, v_act, w_act]
        self.prev_ctrl = np.array([0.0, 0.0]) # [v_cmd, w_cmd] from last step
        self.get_logger().info('Improved MPC Node Started (Exact Integration + Dynamics)')
    def exact_unicycle_model(self, state, v_cmd, w_cmd, dt):
        """
        Integrates State [x, y, theta, v_act, w_act]
        Uses Exact Arc definition for pose AND first-order lag for actuators.
        """
        x, y, theta, v_act, w_act = state
        
        # 1. Update Actuator States (Lag)
        # Simple Euler for the dynamics part is usually fine, or exact exponential decay
        v_next = v_act + self.alpha_v * (v_cmd - v_act) * dt
        w_next = w_act + self.alpha_w * (w_cmd - w_act) * dt
        
        # Use average velocity for pose integration for better stability
        v_avg = (v_act + v_next) / 2.0
        w_avg = (w_act + w_next) / 2.0
        
        # 2. Update Pose (Exact Unicycle)
        if abs(w_avg) < 1e-4:
            # Straight line limit
            x_next = x + v_avg * np.cos(theta) * dt
            y_next = y + v_avg * np.sin(theta) * dt
            theta_next = theta + w_avg * dt # w is tiny
        else:
            # Exact Arc
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            sin_t_w = np.sin(theta + w_avg * dt)
            cos_t_w = np.cos(theta + w_avg * dt)
            
            x_next = x + (v_avg / w_avg) * (sin_t_w - sin_t)
            y_next = y - (v_avg / w_avg) * (cos_t_w - cos_t)
            theta_next = theta + w_avg * dt
            
        return [x_next, y_next, theta_next, v_next, w_next]
    def cost_function(self, u_flat, *args):
        target_x, target_y, start_v, start_w = args
        u = u_flat.reshape((self.horizon, 2))
        
        cost = 0.0
        # Initial State: Robot is at (0,0,0) in its own frame, but has initial velocity
        state = [0.0, 0.0, 0.0, start_v, start_w]
        
        prev_u = np.array([start_v, start_w])
        
        for k in range(self.horizon):
            v_cmd = u[k, 0]
            w_cmd = u[k, 1]
            
            state = self.exact_unicycle_model(state, v_cmd, w_cmd, self.dt)
            x, y, theta, v_act, w_act = state
            
            # --- Path Geometry Cost ---
            # Approximating CTE: Distance to the "Point" target
            # Ideally target has an orientation, but here we just minimize distance + smooth approach
            dist_sq = (x - target_x)**2 + (y - target_y)**2
            
            # Heading Error (Greedy: Point towards target)
            target_angle = math.atan2(target_y - y, target_x - x)
            angle_err = theta - target_angle
            # Normalize angle
            angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi
            
            # Penalties
            w_dist = 10.0
            w_head = 2.0
            w_rate = 5.0  # Smoothness (Jerk)
            
            if dist_sq < 0.05:
                # If very close, stop caring about heading, just stop
                w_head = 0.0
                
            cost += w_dist * dist_sq
            cost += w_head * angle_err**2
            
            # --- Control Rate Penalty (Smoothness) ---
            # Penalize change from PREVIOUS command
            cost += w_rate * ((v_cmd - prev_u[0])**2 + (w_cmd - prev_u[1])**2)
            
            prev_u = [v_cmd, w_cmd]
            
        return cost
    def waypoint_callback(self, msg):
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        
        # State estimation (idealized)
        # We assume current command is roughly current velocity (unless we had Odom feedback)
        # Using last commanded values for continuity
        start_v = self.prev_ctrl[0]
        start_w = self.prev_ctrl[1]
        
        # Initial Guess: Previous solution shifted or just zeros
        u0 = np.zeros(self.horizon * 2)
        
        # --- Bounds (Hard Constraints on Value) ---
        bounds = []
        for _ in range(self.horizon):
            bounds.append((0.0, self.max_v))        # v [0, max]
            bounds.append((-self.max_w, self.max_w)) # w [-max, max]
            
        # --- Constraints (Hard Constraints on Rate) ---
        # Note: SLSQP constraints can be slightly slow in Python, but we use them for correctness
        # Rate Limit: |u_k - u_{k-1}| <= max_acc * dt
        
        # Alternatively, we rely on the Cost Function w_rate penalty for speed
        # For this implementation, we stick to Bounds + Cost Penalty for Rate to ensure 20Hz performance.
        # Adding LinearConstraints for N steps can be heavy.
        
        result = minimize(
            self.cost_function,
            u0,
            args=(tx, ty, start_v, start_w),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-3, 'disp': False, 'maxiter': 20}
        )
        
        if result.success:
            u_opt = result.x.reshape((self.horizon, 2))
            v_cmd = u_opt[0, 0]
            w_cmd = u_opt[0, 1]
            
            # Store for next loop
            self.prev_ctrl = np.array([v_cmd, w_cmd])
            
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'base_link'
            twist_msg.twist.linear.x = float(v_cmd)
            twist_msg.twist.angular.z = float(w_cmd)
            self.publisher.publish(twist_msg)
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MPCNode())
    rclpy.shutdown()
if __name__ == '__main__':
    main()