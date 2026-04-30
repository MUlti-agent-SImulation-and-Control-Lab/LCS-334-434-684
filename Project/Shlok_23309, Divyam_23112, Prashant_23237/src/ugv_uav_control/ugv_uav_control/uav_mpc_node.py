import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import numpy as np
import math
import time
from scipy.optimize import minimize

# Shared utilities
import ugv_uav_control.ugv_model as ugv_model
import ugv_uav_control.view_quality as view_quality

class UAVMPCNode(Node):
    def __init__(self):
        super().__init__('uav_mpc_node')
        
        # === PARAMETERS ===
        self.declare_parameter('z_opt', 5.0)
        self.declare_parameter('z_min', 3.0)
        self.declare_parameter('z_max', 10.0)
        self.declare_parameter('vz_max', 1.0)
        self.declare_parameter('w_alt', 50.0)
        self.declare_parameter('sigma_z', 3.0)
        self.declare_parameter('use_timer', False)
        
        self.z_opt = self.get_parameter('z_opt').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.vz_max = self.get_parameter('vz_max').value
        self.w_alt = self.get_parameter('w_alt').value
        self.sigma_z = self.get_parameter('sigma_z').value
        self.use_timer = self.get_parameter('use_timer').value
        
        # MPC Constants
        self.N = 10  # Hardware budget: <150ms
        self.dt = 0.2
        self.Q_vec = np.linspace(1.0, 2.0, self.N) # Quadratic weighting over horizon
        
        # === STATE VARIABLES ===
        self.uav_pos = None
        self.ugv_mu = None
        self.ugv_traj = None
        
        self.u_prev = np.zeros((self.N, 3))
        self.solve_times = []
        self.total_solves = 0
        
        # === SUBSCRIPTIONS ===
        self.create_subscription(Pose, '/uav_gps_pose', self.uav_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/belief_state', self.belief_callback, 10)
        self.create_subscription(Path, '/ugv_predicted_traj', self.traj_callback, 10)
        self.create_subscription(Bool, '/uav_replan_trigger', self.replan_trigger_callback, 10)
        
        # === PUBLISHERS ===
        self.traj_pub = self.create_publisher(Path, '/uav_nominal_traj', 10)
        
        if self.use_timer:
            self.create_timer(1.0, self.solve_mpc)
            self.get_logger().info('UAV MPC Node: Legacy 1Hz Timer Mode (Condition 1/3)')
        else:
            self.get_logger().info('UAV MPC Node: Optimized 3D Event-Triggered Planner Ready')

    def uav_callback(self, msg):
        self.uav_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

    def belief_callback(self, msg):
        # Extract pose from PoseWithCovarianceStamped
        mu = np.zeros(3)
        mu[0] = msg.pose.pose.position.x
        mu[1] = msg.pose.pose.position.y
        # Yaw extraction ignored for nominal planning (point-mass model)
        self.ugv_mu = mu

    def traj_callback(self, msg):
        # Store future UGV setpoints for vectorized tracking
        traj = []
        for pose in msg.poses:
            traj.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        self.ugv_traj = np.array(traj)

    def replan_trigger_callback(self, msg):
        if msg.data:
            self.solve_mpc()

    def cost_function(self, u_flat, uav_start, ugv_goal):
        u_seq = u_flat.reshape(self.N, 3)
        
        # Vectorized Trajectory Prediction
        # traj_pos[k] = uav_start + sum(u[:k]) * dt
        traj_pos = uav_start + np.cumsum(u_seq, axis=0) * self.dt
        
        # 1. Goal Tracking (Targeting the UGV's predicted state at end of horizon)
        # Using a simplified distance to the last known UGV traj point
        dist_sq = np.sum((traj_pos - ugv_goal)**2, axis=1)
        cost = np.sum(dist_sq * self.Q_vec)
        
        # 2. Optimal Altitude Cost
        alt_err_sq = (traj_pos[:, 2] - self.z_opt)**2
        cost += self.sigma_z * np.sum(alt_err_sq * self.Q_vec)
        
        # 3. Fast Visibility Heuristic (Minimize lateral distance relative to altitude)
        lat_dist_sq = np.sum((traj_pos[:, :2] - ugv_goal[:2])**2, axis=1)
        visibility_cost = lat_dist_sq / (traj_pos[:, 2]**2 + 0.1)
        cost += self.w_alt * np.sum(visibility_cost * self.Q_vec)
        
        # 4. Control Effort
        cost += 0.1 * np.sum(u_seq**2)
        
        return cost

    def solve_mpc(self):
        if self.uav_pos is None or self.ugv_mu is None or self.ugv_traj is None:
            return
            
        start_time = time.time()
        
        # Target is the UGV position at the end of our horizon (or last available)
        idx = min(self.N, len(self.ugv_traj) - 1)
        ugv_goal = self.ugv_traj[idx] if idx >= 0 else self.ugv_mu
        
        # Bounds: [vx, vy, vz]
        b_xy = (-1.5, 1.5)
        b_z = (0.0, 0.0) if self.use_timer else (-self.vz_max, self.vz_max)
        bounds = [b_xy, b_xy, b_z] * self.N
        
        # Warm start
        u0 = np.roll(self.u_prev, -1, axis=0)
        u0[-1] = [0.0, 0.0, 0.0]
        
        res = minimize(
            self.cost_function, 
            u0.flatten(), 
            args=(self.uav_pos, ugv_goal),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 15, 'ftol': 1e-3}
        )
        
        solve_duration = time.time() - start_time
        self.solve_times.append(solve_duration)
        self.total_solves += 1
        
        if res.success:
            u_opt = res.x.reshape(self.N, 3)
            self.u_prev = u_opt
            self.publish_nominal_path(u_opt)
        else:
            self.get_logger().warn(f'MPC Solve Failed: {res.message}')

        if self.total_solves % 10 == 0:
            avg_ms = np.mean(self.solve_times[-10:]) * 1000
            self.get_logger().info(f'[MPC] PHASE 2 SUMMARY | Solve: {avg_ms:.1f}ms | Goal Dist: {np.linalg.norm(self.uav_pos - ugv_goal):.2f}m')

    def publish_nominal_path(self, u_opt):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'world'
        
        curr_pos = self.uav_pos.copy()
        
        # Point 0: Current Position
        p0 = PoseStamped()
        p0.header = path.header
        p0.pose.position.x, p0.pose.position.y, p0.pose.position.z = map(float, curr_pos)
        path.poses.append(p0)
        
        for k in range(self.N):
            curr_pos += u_opt[k] * self.dt
            pk = PoseStamped()
            # Approximate timestamping for ancillary interpolation
            ts = self.get_clock().now().nanoseconds + int((k+1) * self.dt * 1e9)
            pk.header.stamp.sec = int(ts // 1e9)
            pk.header.stamp.nanosec = int(ts % 1e9)
            pk.header.frame_id = 'world'
            pk.pose.position.x, pk.pose.position.y, pk.pose.position.z = map(float, curr_pos)
            path.poses.append(pk)
            
        self.traj_pub.publish(path)

def main(args=None):
    rclpy.init(args=args)
    node = UAVMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
