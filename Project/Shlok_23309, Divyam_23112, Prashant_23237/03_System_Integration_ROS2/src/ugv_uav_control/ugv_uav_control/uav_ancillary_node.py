"""
Phase 3: Mission Integration & Validation (ROS 2)
Component: Adaptive Tube-MPC Ancillary Node
Purpose: Executes the high-dynamic UAV tracking policy. 
         Adapts UAV altitude and pose based on UGV curvature and estimation 
         uncertainty to maintain the 'Visual Tube' (Field of View).
Inputs: /model/uav/pose, /belief_state
Outputs: /model/uav/cmd_vel (UAV Steering)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import numpy as np
import math
import os
from ament_index_python.packages import get_package_share_directory

class UAVAncillaryNode(Node):
    def __init__(self):
        super().__init__('uav_ancillary_node')
        
        # === ROS PARAMETERS ===
        self.declare_parameter('alpha_min', 0.5)
        self.declare_parameter('alpha_max', 2.0)
        self.declare_parameter('tr_min', 0.005)
        self.declare_parameter('tr_max', 0.15)
        self.declare_parameter('tr_spike_threshold', 0.12)
        self.declare_parameter('kappa_thresh', 0.4)
        self.declare_parameter('drift_threshold', 0.07)
        self.declare_parameter('T_min', 3.0)
        
        self.alpha_min = self.get_parameter('alpha_min').value
        self.alpha_max = self.get_parameter('alpha_max').value
        self.tr_min = self.get_parameter('tr_min').value
        self.tr_max = self.get_parameter('tr_max').value
        self.tr_spike_threshold = self.get_parameter('tr_spike_threshold').value
        self.kappa_thresh = self.get_parameter('kappa_thresh').value
        self.drift_threshold = self.get_parameter('drift_threshold').value
        self.T_min = self.get_parameter('T_min').value
        
        # === TUBE PARAMS ===
        # Load from scratch directory where we computed them
        params_path = "/home/shlok-mehndiratta/.gemini/antigravity/scratch/tube_params.npz"
        if os.path.exists(params_path):
            data = np.load(params_path)
            self.K = data['K']
            self.P_0 = data['P_0']
            self.get_logger().info(f"Loaded tube params: K={self.K.diagonal()}, P_0_eig={np.linalg.eigvals(self.P_0)[0]:.2f}")
        else:
            self.get_logger().error(f"Tube params NOT FOUND at {params_path}!")
            # Fallback defaults if file missing (K=2, P=6.86)
            self.K = np.eye(3) * 2.0
            self.P_0 = np.eye(3) * 6.86
            
        # === STATE VARIABLES ===
        self.uav_pos = None      # Actual state
        self.nominal_traj = None # Path (N+1 poses)
        self.nominal_controls = None # Corresponding u_bar
        self.ugv_traj = None     # For curvature detection
        self.last_tr_sigma = 0.0
        self.tr_sigma_history = [] # For spike detection
        self.last_replan_time = self.get_clock().now()
        self.last_trigger_level = ""
        self.trigger_lockouts = {"1b": 0.0, "2": 0.0, "3": 0.0} # Last trigger times (seconds)
        
        # === SUBSCRIPTIONS ===
        self.create_subscription(Pose, '/model/uav/pose', self.uav_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/belief_state', self.belief_callback, 10)
        self.create_subscription(Path, '/ugv_predicted_traj', self.ugv_traj_callback, 10)
        self.create_subscription(Path, '/uav_nominal_traj', self.nominal_traj_callback, 10)
        
        # === PUBLISHERS ===
        self.cmd_pub = self.create_publisher(Twist, '/model/uav/cmd_vel', 10)
        self.replan_pub = self.create_publisher(Bool, '/uav_replan_trigger', 10)
        
        # === TIMER (20Hz) ===
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.state = 'TAKEOFF'
        self.takeoff_height = 5.0
        self.takeoff_velocity = 0.5
        
        self.get_logger().info('UAV Ancillary Node Started (20Hz)')

    def uav_callback(self, msg):
        self.uav_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

    def belief_callback(self, msg):
        # tr(Sigma) from 6x6 cov (x,y,z,roll,pitch,yaw)
        cov = np.array(msg.pose.covariance).reshape(6, 6)
        # We care about horizontal x,y for the scaling tr_sigma
        self.last_tr_sigma = cov[0,0] + cov[1,1]
        
        # History for spike detection (0.5s = 10 samples at 20Hz if synced, 
        # but belief is 20Hz so we store last 10)
        now = self.get_clock().now().nanoseconds / 1e9
        self.tr_sigma_history.append((now, self.last_tr_sigma))
        if len(self.tr_sigma_history) > 20: # keep ~1s
            self.tr_sigma_history.pop(0)

    def ugv_traj_callback(self, msg):
        self.ugv_traj = msg

    def nominal_traj_callback(self, msg):
        self.nominal_traj = msg
        self.last_replan_time = self.get_clock().now()
        # Initial trigger to get first traj? MPC should publish first one anyway.
        
    def get_nominal_state(self, now_ts):
        """Linearly interpolates nominal trajectory based on current timestamp."""
        if self.nominal_traj is None or len(self.nominal_traj.poses) < 2:
            return None, None
        
        # Trajectory assumes steps of dt=0.3s
        # Start time is header timestamp of nominal_traj
        t0 = rclpy.time.Time.from_msg(self.nominal_traj.header.stamp).nanoseconds / 1e9
        idx = int((now_ts - t0) / 0.3)
        
        if idx < 0:
            p = self.nominal_traj.poses[0].pose.position
            return np.array([p.x, p.y, p.z]), np.array([0.0, 0.0, 0.0])
        elif idx >= len(self.nominal_traj.poses) - 1:
            # Past end of plan, hold last
            p = self.nominal_traj.poses[-1].pose.position
            return np.array([p.x, p.y, p.z]), np.array([0.0, 0.0, 0.0])
        
        # Simple interpolation
        p1 = self.nominal_traj.poses[idx].pose.position
        p2 = self.nominal_traj.poses[idx+1].pose.position
        ratio = ((now_ts - t0) % 0.3) / 0.3
        
        x_bar = np.array([
            p1.x + ratio * (p2.x - p1.x),
            p1.y + ratio * (p2.y - p1.y),
            p1.z + ratio * (p2.z - p1.z)
        ])
        
        # Nominal velocity bar (finite difference of traj steps)
        u_bar = np.array([
            (p2.x - p1.x) / 0.3,
            (p2.y - p1.y) / 0.3,
            (p2.z - p1.z) / 0.3
        ])
        
        return x_bar, u_bar

    def timer_callback(self):
        if self.uav_pos is None:
            return
            
        now_ts = self.get_clock().now().nanoseconds / 1e9
        
        # 0. State Machine (Takeoff/Tracking)
        if self.state == 'TAKEOFF':
            if self.uav_pos[2] < (self.takeoff_height - 0.2):
                msg = Twist()
                msg.linear.z = self.takeoff_velocity
                self.cmd_pub.publish(msg)
                return
            else:
                self.get_logger().info("[ANC] Takeoff Complete. Switching to TRACKING.")
                self.state = 'TRACKING'

        # 1. Initial Bootstrap Trigger (if no plan exists yet)
        if self.nominal_traj is None:
            if not hasattr(self, 'bootstrap_sent'):
                self.get_logger().info("[ANC] Bootstrapping first nominal plan...")
                out_msg = Bool(); out_msg.data = True
                self.replan_pub.publish(out_msg)
                self.bootstrap_sent = True
            return
        
        # 2. Control Logic
        x_bar, u_bar = self.get_nominal_state(now_ts)
        if x_bar is None: return
        
        e_k = self.uav_pos - x_bar
        
        # Tube Scaling
        alpha_k = self.alpha_min + (self.alpha_max - self.alpha_min) * \
                  np.clip((self.last_tr_sigma - self.tr_min)/(self.tr_max - self.tr_min), 0, 1)
        
        # Boundary Check
        inside_tube = (e_k.T @ self.P_0 @ e_k) <= (alpha_k**2)
        
        # Ancillary Command (u_anc = u_bar - K*e)
        u_anc = u_bar - self.K @ e_k
        u_anc = np.clip(u_anc, -1.0, 1.0)
        
        msg = Twist()
        msg.linear.x = float(u_anc[0]); msg.linear.y = float(u_anc[1]); msg.linear.z = float(u_anc[2])
        self.cmd_pub.publish(msg)
        
        # 3. Event Trigger Logic
        trigger = False
        replan_reason = ""
        level = ""
        
        # Level 1a - Tube Exit (Critical)
        if not inside_tube:
            trigger = True; replan_reason = "Tube Exit"; level = "1a"
            
        # Lockout check for non-critical levels
        can_trigger_1b = (now_ts - self.trigger_lockouts["1b"]) > 1.0
        can_trigger_2 = (now_ts - self.trigger_lockouts["2"]) > 1.0
        can_trigger_3 = (now_ts - self.trigger_lockouts["3"]) > 1.0
        
        # Level 1b - ArUco Spike
        if not trigger and can_trigger_1b:
            if self.last_tr_sigma > self.tr_spike_threshold:
                past_05 = [v for t, v in self.tr_sigma_history if (now_ts - t) >= 0.45 and (now_ts - t) <= 0.55]
                if past_05 and self.last_tr_sigma > (1.5 * past_05[0]):
                    trigger = True; replan_reason = "ArUco Spike"; level = "1b"
                    self.trigger_lockouts["1b"] = now_ts

        # Level 2 - Curvature Anticipation
        max_kappa = 0.0
        if not trigger and can_trigger_2 and self.ugv_traj is not None:
            poses = self.ugv_traj.poses
            for i in range(min(2, len(poses)-1)):
                p1 = poses[i].pose.position; p2 = poses[i+1].pose.position
                q1 = poses[i].pose.orientation; q2 = poses[i+1].pose.orientation
                y1 = math.atan2(2*(q1.w*q1.z + q1.x*q1.y), 1-2*(q1.y*q1.y + q1.z*q1.z))
                y2 = math.atan2(2*(q2.w*q2.z + q2.x*q2.y), 1-2*(q2.y*q2.y + q2.z*q2.z))
                dth = abs(math.atan2(math.sin(y2-y1), math.cos(y2-y1)))
                ds = math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2)
                kappa = dth / (ds + 1e-6)
                max_kappa = max(max_kappa, kappa)
            if max_kappa > self.kappa_thresh:
                trigger = True; replan_reason = f"Curvature ({max_kappa:.2f})"; level = "2"
                self.trigger_lockouts["2"] = now_ts
                
        # Level 3 - Drift Maintenance
        if not trigger and can_trigger_3:
            time_since_replan = (self.get_clock().now() - self.last_replan_time).nanoseconds / 1e9
            if self.last_tr_sigma > self.drift_threshold and time_since_replan > self.T_min:
                trigger = True; replan_reason = "Drift Maintenance"; level = "3"
                self.trigger_lockouts["3"] = now_ts
                
        if trigger:
            self.get_logger().info(f"[TRIGGER] Level {level}: {replan_reason} | tr(Sigma): {self.last_tr_sigma:.4f} | UAV_Z: {self.uav_pos[2]:.2f}")
            out_msg = Bool(); out_msg.data = True
            self.replan_pub.publish(out_msg)
        elif (now_ts % 2.0) < 0.05: # Heartbeat every 2s
            e_norm = np.linalg.norm(e_k)
            self.get_logger().info(f"[ANC_HB] State: {self.state} | e_norm: {e_norm:.3f} | Alpha: {alpha_k:.2f} | Inside: {inside_tube}")

def main(args=None):
    rclpy.init(args=args)
    node = UAVAncillaryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
