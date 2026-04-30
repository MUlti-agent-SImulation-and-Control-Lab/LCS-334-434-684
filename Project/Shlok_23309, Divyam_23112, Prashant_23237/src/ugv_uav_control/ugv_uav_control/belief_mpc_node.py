import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from scipy.optimize import minimize
from ugv_uav_control import ugv_model
import numpy as np
import math

class BeliefMPCNode(Node):
    def __init__(self):
        super().__init__('belief_mpc_node')
        
        # === SUBSCRIPTIONS ===
        self.create_subscription(PoseStamped, '/waypoints', self.waypoint_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/belief_state', self.belief_callback, 10)
        self.create_subscription(Pose, '/uav_gps_pose', self.uav_callback, 10)
        
        # === PUBLISHERS ===
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.traj_pub = self.create_publisher(Path, '/ugv_predicted_traj', 10)
        
        # === MPC PARAMETERS ===
        self.horizon = 5
        self.dt = 0.2
        self.max_v = 0.22
        self.max_w = 2.0
        
        # Actuator dynamics
        self.alpha_v = 2.0
        self.alpha_w = 5.0
        
        # Process noise (same as EKF)
        self.Q = np.diag([1e-4, 1e-4, 4e-4, 1e-4, 2.5e-3])
        
        # === BELIEF STATE (from EKF) ===
        self.mu = np.zeros(5)  # [x, y, θ, v, ω]
        self.Sigma = np.eye(5) * 0.1
        self.world_pos = np.zeros(3)  # [x, y, theta] in world frame from EKF
        
        # === COST WEIGHTS ===
        self.w_dist = 10.0       # Distance to target
        self.w_head = 2.0        # Heading error
        self.w_rate = 5.0        # Control rate (smoothness)
        self.w_uncertainty = 1.0 # λ: Uncertainty penalty weight
        
        self.prev_ctrl = np.array([0.0, 0.0])
        
        self.get_logger().info('Belief-Space MPC Node Started (λ = {:.2f})'.format(self.w_uncertainty))

        # UAV parameters (static for now)
        self.uav_x = 0.0
        self.uav_y = 0.0
        self.uav_z = 5.0
        self.fov_angle = 1.047  # 60 degrees

        # Compute FoV radius

        # Observability smoothing parameter
        self.gamma_k = 5.0  # Sigmoid sharpness

        # Observation matrices (for when visible)
        self.H_obs = np.eye(5)
        self.R_obs_base = np.diag([0.0025, 0.0025, 0.01, 0.0004, 0.0025])
        
        # UAV state for dynamic R_k
        self.uav_pos = np.array([0.0, 0.0, 5.0])
        self.last_log_time_mpc = self.get_clock().now()
        self.last_log_time_sigma = self.get_clock().now()
        self.fov_angle = 1.047

    
    
    # =========================================
    # BELIEF-SPACE COST FUNCTION
    # =========================================
    def cost_function(self, u_flat, *args):
        target_x, target_y, start_mu, start_Sigma, start_v, start_w, world_pos = args
        u = u_flat.reshape((self.horizon, 2))
        
        cost = 0.0
        mu = start_mu.copy()
        Sigma = start_Sigma.copy()
        prev_u = np.array([start_v, start_w])
        
        for k in range(self.horizon):
            v_cmd = u[k, 0]
            w_cmd = u[k, 1]
            
            # Mean propagation
            mu = ugv_model.process_model(mu, [v_cmd, w_cmd], self.dt, self.alpha_v, self.alpha_w)
            
            # Covariance prediction
            F = ugv_model.compute_jacobian_F(mu, self.dt, self.alpha_v, self.alpha_w)
            Sigma_pred = F @ Sigma @ F.T + self.Q
            
            # ========================================
            # FoV-DEPENDENT COVARIANCE UPDATE (NEW)
            # ========================================
            # Current predicted position in WORLD frame
            world_x = world_pos[0] + mu[0] * np.cos(world_pos[2]) - mu[1] * np.sin(world_pos[2])
            world_y = world_pos[1] + mu[0] * np.sin(world_pos[2]) + mu[1] * np.cos(world_pos[2])
            world_mu = np.array([world_x, world_y, 0, 0, 0])  # Only need x, y for FoV
            
            gamma, d = self.compute_observability(world_mu)
            
            # Blend between no-observation and observation
            if gamma > 0.01:
                # Use shared logic for R_k and Sigma_obs
                R_k, q_k = ugv_model.get_dynamic_R(
                    self.uav_pos, world_mu[:2], self.R_obs_base
                )
                Sigma_obs = ugv_model.kalman_update(Sigma_pred, self.H_obs, R_k)
                Sigma = (1 - gamma) * Sigma_pred + gamma * Sigma_obs
                
                # Throttled logging (every 5s) - only for first horizon step to avoid spam
                if k == 0:
                    now = self.get_clock().now()
                    if (now - self.last_log_time_mpc).nanoseconds > 5e9:
                        self.get_logger().info(
                            f'[MPC] Dynamic R_k: q={q_k:.3f}, tr(R)={np.trace(R_k):.4f}'
                        )
                        self.last_log_time_mpc = now
            else:
                Sigma = Sigma_pred
            
            # ========================================
            # TASK COST
            # ========================================
            x, y, theta = mu[0], mu[1], mu[2]
            
            dist_sq = (x - target_x)**2 + (y - target_y)**2
            target_angle = math.atan2(target_y - y, target_x - x)
            angle_err = ugv_model.wrap_angle(theta - target_angle)
            w_head_k = self.w_head if dist_sq > 0.05 else 0.0
            rate_penalty = (v_cmd - prev_u[0])**2 + (w_cmd - prev_u[1])**2
            
            # ========================================
            # UNCERTAINTY + FoV COST
            # ========================================
            uncertainty_cost = np.trace(Sigma)
            
            # Optional: Add explicit FoV-exit penalty
            r_fov_curr = self.uav_pos[2] * np.tan(self.fov_angle / 2.0)
            fov_penalty = max(0, d - r_fov_curr)**2  # Penalty for leaving FoV
            
            # ========================================
            # TOTAL COST
            # ========================================
            cost += self.w_dist * dist_sq
            cost += w_head_k * angle_err**2
            cost += self.w_rate * rate_penalty
            cost += self.w_uncertainty * uncertainty_cost
            cost += 2.0 * fov_penalty  # Additional FoV penalty
            
            prev_u = [v_cmd, w_cmd]
        
        return cost
    
    # =========================================
    # CALLBACKS
    # =========================================
    def belief_callback(self, msg):
        """Receive belief state from EKF."""
        self.mu[0] = msg.pose.pose.position.x
        self.mu[1] = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.mu[2] = math.atan2(siny_cosp, cosy_cosp)
        
        # Store world-frame position AFTER yaw is correctly computed
        # This is used by the FoV observability check inside the optimizer
        self.world_pos = np.array([self.mu[0], self.mu[1], self.mu[2]])
        
        # Extract covariance (6x6 in message, we need 5x5)
        cov_flat = np.array(msg.pose.covariance).reshape((6, 6))
        self.Sigma[0, 0] = cov_flat[0, 0]
        self.Sigma[0, 1] = cov_flat[0, 1]
        self.Sigma[1, 0] = cov_flat[1, 0]
        self.Sigma[1, 1] = cov_flat[1, 1]
        self.Sigma[2, 2] = cov_flat[5, 5]  # yaw variance
        self.Sigma[0, 2] = cov_flat[0, 5]
        self.Sigma[2, 0] = cov_flat[5, 0]
        self.Sigma[1, 2] = cov_flat[1, 5]
        self.Sigma[2, 1] = cov_flat[5, 1]

        # Log Sigma update confirmation (throttled)
        now = self.get_clock().now()
        if (now - self.last_log_time_sigma).nanoseconds > 5e9:
            self.get_logger().info(f'[MPC] Sigma updated. Trace: {np.trace(self.Sigma):.6f}')
            self.last_log_time_sigma = now
        
        # Velocity states (keep from last control command)
        # self.mu[3], self.mu[4] are updated via control commands
        
    def uav_callback(self, msg):
        # Handle both Pose and PoseStamped for robustness
        if hasattr(msg, 'pose'):
            self.uav_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        else:
            self.uav_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        
        # Keep uav_x/y/z legacy fields updated if referenced
        self.uav_x, self.uav_y, self.uav_z = self.uav_pos
    
    def waypoint_callback(self, msg):
        """Optimize control given target waypoint in ROBOT FRAME."""
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        
        # Current velocities (from last command)
        start_v = self.prev_ctrl[0]
        start_w = self.prev_ctrl[1]
        
        # In robot frame, robot starts at origin facing forward
        # mu is [x, y, theta, v, omega] in ROBOT FRAME
        start_mu = np.array([0.0, 0.0, 0.0, start_v, start_w])
        
        # Use uncertainty from EKF (still valid)
        start_Sigma = self.Sigma.copy()
        
        # Initial guess
        u0 = np.zeros(self.horizon * 2)
        
        # Bounds
        bounds = []
        for _ in range(self.horizon):
            bounds.append((0.0, self.max_v))
            bounds.append((-self.max_w, self.max_w))
        
        # Optimize
        result = minimize(
            self.cost_function,
            u0,
            args=(tx, ty, start_mu, start_Sigma, start_v, start_w, self.world_pos.copy()),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-3, 'maxiter': 20}
        )
        
        if result.success:
            u_opt = result.x.reshape((self.horizon, 2))
            v_cmd = u_opt[0, 0]
            w_cmd = u_opt[0, 1]
            
            # Update internal state
            self.prev_ctrl = np.array([v_cmd, w_cmd])
            
            # Publish control
            cmd_msg = TwistStamped()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.twist.linear.x = v_cmd
            cmd_msg.twist.angular.z = w_cmd
            self.publisher.publish(cmd_msg)

            # Publish predicted trajectory for UAV
            self.publish_predicted_trajectory(u_opt, start_mu)
        else:
            # Fallback if optimization fails
            v_cmd = 0.1
            w_cmd = 0.0
        
        self.prev_ctrl = np.array([v_cmd, w_cmd])
        
        # Publish command
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        twist_msg.twist.linear.x = float(v_cmd)
        twist_msg.twist.angular.z = float(w_cmd)
        self.publisher.publish(twist_msg)


    def compute_observability(self, mu):
        """
        Compute smooth observability based on UGV position relative to UAV FoV.
        Returns γ ∈ [0, 1]
        """
        # Distance from UGV to UAV ground projection
        dx = mu[0] - self.uav_x
        dy = mu[1] - self.uav_y
        d = np.sqrt(dx**2 + dy**2)
        
        # Smooth sigmoid: γ = 1 / (1 + exp(k * (d - r)))
        r_fov_curr = self.uav_pos[2] * np.tan(self.fov_angle / 2.0)
        gamma = 1.0 / (1.0 + np.exp(self.gamma_k * (d - r_fov_curr)))
        
        return gamma, d

    def publish_predicted_trajectory(self, u_opt, start_mu):
        """Perform final rollout and publish as Path message."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "world"
        
        mu = start_mu.copy()
        now = self.get_clock().now()
        
        # Origin for world frame transform
        x0, y0, th0 = self.world_pos
        
        for k in range(self.horizon + 1):
            # Transform to world frame
            # mu[0,1] are relative x,y; mu[2] is relative theta
            wx = x0 + mu[0] * math.cos(th0) - mu[1] * math.sin(th0)
            wy = y0 + mu[0] * math.sin(th0) + mu[1] * math.cos(th0)
            wth = ugv_model.wrap_angle(th0 + mu[2])
            
            pose = PoseStamped()
            # Time offset for each step
            step_time = now.nanoseconds + int(k * self.dt * 1e9)
            pose.header.stamp = rclpy.time.Time(nanoseconds=step_time).to_msg()
            pose.header.frame_id = "world"
            
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            
            q = self._quaternion_from_euler(0, 0, wth)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            
            path_msg.poses.append(pose)
            
            if k < self.horizon:
                # Predict next state using central logic
                mu = ugv_model.process_model(mu, u_opt[k], self.dt, self.alpha_v, self.alpha_w)

        self.traj_pub.publish(path_msg)

    def _quaternion_from_euler(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        q = [0.0] * 4
        q[0] = cy * cp * sr - sy * sp * cr
        q[1] = sy * cp * sr + cy * sp * cr
        q[2] = sy * cp * cr - cy * sp * sr
        q[3] = cy * cp * cr + sy * sp * sr
        return q



def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BeliefMPCNode())
    rclpy.shutdown()
if __name__ == '__main__':
    main()