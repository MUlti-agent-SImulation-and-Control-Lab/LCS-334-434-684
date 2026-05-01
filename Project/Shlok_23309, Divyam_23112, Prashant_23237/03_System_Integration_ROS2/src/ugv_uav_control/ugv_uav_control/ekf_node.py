"""
Phase 3: Mission Integration & Validation (ROS 2)
Component: Extended Kalman Filter (EKF) Node
Purpose: Fuses noisy ArUco detections and UAV GPS poses to estimate 
         the full 5-state belief [x, y, theta, v, omega] of the UGV.
         Provides the 'Belief State' required for the Information-Aware MPC.
Inputs: /aruco_pose, /uav_gps_pose
Outputs: /belief_state (PoseWithCovarianceStamped)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from ugv_uav_control import ugv_model
import numpy as np
import math
    
def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        cy * cp * sr - sy * sp * cr,  # x
        sy * cp * sr + cy * sp * cr,  # y
        sy * cp * cr - cy * sp * sr,  # z
        cy * cp * cr + sy * sp * sr   # w
    ]

def euler_from_quaternion(q):
    """Extract yaw from quaternion [x, y, z, w]."""
    siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
    cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    return math.atan2(siny_cosp, cosy_cosp)

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        
        # === STATE ===
        # [x, y, θ, v, ω]
        self.mu = np.zeros(5)
        self.Sigma = np.diag([0.1, 0.1, 0.1, 0.01, 0.01])
        
        # === PARAMETERS ===
        self.dt = 0.05  # 20 Hz prediction rate
        self.alpha_v = 2.0   # 1/τ_v (τ_v = 0.5s)
        self.alpha_w = 5.0   # 1/τ_w (τ_w = 0.2s)
        self.start_time_ekf = self.get_clock().now()
        
        # Process Noise Q
        self.Q = np.diag([1e-4, 1e-4, 4e-4, 1e-4, 2.5e-3])
        
        # Measurement Noise
        self.R_full = np.diag([0.0025, 0.0025, 0.01, 0.0004, 0.0025])  # ArUco + Odom
        self.R_odom = np.diag([0.0004, 0.0025])  # Odom only
        
        # Control cache
        self.last_cmd = np.array([0.0, 0.0])
        self.last_aruco_time = self.get_clock().now()
        self.aruco_timeout = 2.0  # seconds
        
        # UAV state for dynamic R_k
        self.uav_pos = np.array([0.0, 0.0, 5.0])
        self.last_log_time_ekf = self.get_clock().now()

        # Sanity Check: ArUco Blocking
        self.declare_parameter('block_aruco', False)
        self.declare_parameter('block_aruco_start', 25.0)
        self.declare_parameter('block_aruco_end', 30.0)
        
        # === SUBSCRIBERS ===
        self.create_subscription(TwistStamped, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/aruco_pose', self.aruco_callback, 10)
        self.create_subscription(Pose, '/uav_gps_pose', self.uav_callback, 10)
        
        # === PUBLISHER ===
        self.belief_pub = self.create_publisher(PoseWithCovarianceStamped, '/belief_state', 10)
        
        # === TIMER for prediction ===
        self.create_timer(self.dt, self.prediction_step)
        
        self.get_logger().info('EKF Node Started (5-state: x, y, θ, v, ω)')
    
    # =========================================
    # PROCESS MODEL
    # =========================================
    # =========================================
    # EKF STEPS
    # =========================================
    
    # =========================================
    # EKF STEPS
    # =========================================
    def prediction_step(self):
        """
        Timer callback: Predict belief forward using last control.
        """
        # Predict mean
        self.mu = ugv_model.process_model(self.mu, self.last_cmd, self.dt, self.alpha_v, self.alpha_w)
        
        # Predict covariance
        F = ugv_model.compute_jacobian_F(self.mu, self.dt, self.alpha_v, self.alpha_w)
        self.Sigma = F @ self.Sigma @ F.T + self.Q
        
        # Publish belief
        self.publish_belief()
    
    def update_step(self, z, H, R):
        """
        Generic EKF update step.
        z: measurement vector
        H: measurement Jacobian
        R: measurement noise covariance
        """
        # Innovation
        z_pred = H @ self.mu
        y = z - z_pred
        
        # Handle angle wrapping for θ (index 2)
        if H.shape[0] >= 3 and H[2, 2] == 1.0:
            y[2] = ugv_model.wrap_angle(y[2])
        
        # Kalman gain
        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        
        # State update
        self.mu = self.mu + K @ y
        self.mu[2] = ugv_model.wrap_angle(self.mu[2])  # Normalize θ
        
        # Covariance update (Joseph form)
        self.Sigma = ugv_model.kalman_update(self.Sigma, H, R)
    
    def cmd_callback(self, msg):
        """Store latest control command."""
        self.last_cmd = np.array([
            msg.twist.linear.x,
            msg.twist.angular.z
        ])

    def uav_callback(self, msg):
        """Update UAV position for view quality calculation."""
        self.uav_pos[0] = msg.position.x
        self.uav_pos[1] = msg.position.y
        self.uav_pos[2] = msg.position.z
    
    def odom_callback(self, msg):
        """
        Update with odometry measurement (v, ω).
        """
        z = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ])
        
        H = np.array([
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ], dtype=float)
        
        self.update_step(z, H, self.R_odom)
    def aruco_callback(self, msg):
        """
        Update with ArUco pose measurement (x, y, θ).
        Full 5-state update: pose from ArUco, velocity from current estimate.
        """
        # --- Sanity Check: Blocking Logic ---
        if self.get_parameter('block_aruco').value:
            elapsed = (self.get_clock().now() - self.start_time_ekf).nanoseconds / 1e9
            t_start = self.get_parameter('block_aruco_start').value
            t_end = self.get_parameter('block_aruco_end').value
            if elapsed >= t_start and elapsed <= t_end:
                return

        x_aruco = msg.pose.position.x
        y_aruco = msg.pose.position.y
        q = msg.pose.orientation
        theta_aruco = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Full measurement: [x, y, θ, v, ω]
        # We measure pose directly; velocity inherited from prediction
        z = np.array([x_aruco, y_aruco, theta_aruco, self.mu[3], self.mu[4]])
        
        H = np.eye(5)
        
        # --- DYNAMIC R_k ---
        R_k, q_k = ugv_model.get_dynamic_R(
            self.uav_pos, [x_aruco, y_aruco], self.R_full
        )
        
        self.update_step(z, H, R_k)
        
        # Throttled logging (every 5s)
        now = self.get_clock().now()
        if (now - self.last_log_time_ekf).nanoseconds > 5e9:
            self.get_logger().info(
                f'[EKF] Dynamic R_k: q={q_k:.3f}, tr(R)={np.trace(R_k):.4f}'
            )
            self.last_log_time_ekf = now
        
        self.last_aruco_time = self.get_clock().now()
        self.get_logger().debug(f'ArUco update: x={x_aruco:.2f}, y={y_aruco:.2f}, θ={theta_aruco:.2f}')
    
    # =========================================
    # PUBLISHING
    # =========================================
    def publish_belief(self):
        """
        Publish belief as PoseWithCovarianceStamped.
        """
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        
        # Pose (x, y, θ)
        msg.pose.pose.position.x = float(self.mu[0])
        msg.pose.pose.position.y = float(self.mu[1])
        msg.pose.pose.position.z = 0.0
        
        q = quaternion_from_euler(0, 0, self.mu[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        
        # Covariance: 6x6 matrix (x, y, z, roll, pitch, yaw)
        # We populate the relevant entries from our 5x5 Sigma
        cov = np.zeros((6, 6))
        cov[0, 0] = self.Sigma[0, 0]  # x
        cov[1, 1] = self.Sigma[1, 1]  # y
        cov[2, 2] = self.Sigma[3, 3]  # v (mapped to z)
        cov[3, 3] = self.Sigma[4, 4]  # w (mapped to roll)
        cov[5, 5] = self.Sigma[2, 2]  # theta (yaw)
        
        # Off-diagonals for pose only
        cov[0, 1] = self.Sigma[0, 1]; cov[1, 0] = self.Sigma[1, 0]
        cov[0, 5] = self.Sigma[0, 2]; cov[5, 0] = self.Sigma[2, 0]
        cov[1, 5] = self.Sigma[1, 2]; cov[5, 1] = self.Sigma[2, 1]
        
        msg.pose.covariance = cov.flatten().tolist()
        
        self.belief_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()