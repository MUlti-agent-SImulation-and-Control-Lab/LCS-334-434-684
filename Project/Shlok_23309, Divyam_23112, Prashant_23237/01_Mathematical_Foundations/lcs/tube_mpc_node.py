#!/usr/bin/env python3
"""
Tube MPC Controller Node for ROS2/Gazebo.

Tube MPC with auxiliary controller (solves at every step).
Uses tightened constraints and local feedback gain K.
Publishes acceleration commands.

Subscriptions:
    /drone/odom (nav_msgs/Odometry): Current drone state
    /waypoint (geometry_msgs/Point): Target position waypoint

Publications:
    /drone/cmd_accel (geometry_msgs/Accel): Commanded acceleration
    /drone/mpc_status (std_msgs/String): Solver status
"""

from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point, Accel
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
from typing import List

from lcs.mpc_solvers import TubeMPC


class TubeMPCNode(Node):
    """ROS2 node for Tube MPC control."""
    
    def __init__(self):
        super().__init__('tube_mpc_node')
        
        # Default data path
        base_dir = Path(__file__).resolve().parent.parent
        default_tube_path = str(base_dir / 'simulations' / 'tube_data.npz')
        
        # Parameters
        self.declare_parameter('horizon', 20)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('control_rate', 20.0)  # Hz
        self.declare_parameter('tube_data_path', default_tube_path)
        self.declare_parameter('waypoint_threshold', 0.1)
        
        # Get parameters
        horizon = self.get_parameter('horizon').value
        dt = self.get_parameter('dt').value
        control_rate = self.get_parameter('control_rate').value
        tube_data_path = self.get_parameter('tube_data_path').value
        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        
        # Initialize MPC solver
        self.get_logger().info(f'Initializing Tube MPC: horizon={horizon}, dt={dt}')
        self.mpc = TubeMPC(
            horizon=horizon,
            dt=dt,
            tube_data_path=tube_data_path
        )
        self.mpc.reset()
        
        # State and waypoint tracking
        self.current_state = None
        self.waypoints: List[np.ndarray] = []
        self.current_wp_idx = 0
        self.default_hover = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        
        # QoS profile
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, qos)
        self.create_subscription(Point, '/waypoint', self._waypoint_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Accel, '/drone/cmd_accel', 10)
        self.status_pub = self.create_publisher(String, '/drone/mpc_status', 10)
        
        # Control timer
        timer_period = 1.0 / control_rate
        self.timer = self.create_timer(timer_period, self._control_loop)
        
        self.get_logger().info('Tube MPC Node initialized')
    
    def _odom_callback(self, msg: Odometry):
        """Update current state from odometry."""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        
        self.current_state = np.array([px, py, pz, vx, vy, vz])
    
    def _waypoint_callback(self, msg: Point):
        """Add new waypoint to queue."""
        waypoint = np.array([msg.x, msg.y, msg.z, 0.0, 0.0, 0.0])
        self.waypoints.append(waypoint)
        self.get_logger().info(f'Added waypoint: [{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}] '
                              f'(Queue: {len(self.waypoints)})')
    
    def _get_current_reference(self) -> np.ndarray:
        """Get current reference state based on waypoint progression."""
        if not self.waypoints:
            return self.default_hover
        
        # Check if current waypoint is reached
        if self.current_state is not None and self.current_wp_idx < len(self.waypoints):
            current_wp = self.waypoints[self.current_wp_idx]
            dist = np.linalg.norm(self.current_state[0:3] - current_wp[0:3])
            
            if dist < self.waypoint_threshold:
                self.current_wp_idx += 1
                if self.current_wp_idx < len(self.waypoints):
                    self.get_logger().info(f'Waypoint reached. Moving to next ({self.current_wp_idx + 1}/{len(self.waypoints)})')
                else:
                    self.get_logger().info('All waypoints completed. Hovering at final position.')
        
        # Return current or last waypoint
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        elif self.waypoints:
            return self.waypoints[-1]
        else:
            return self.default_hover
    
    def _control_loop(self):
        """Main control loop."""
        if self.current_state is None:
            self.get_logger().warn('No odometry received yet', throttle_duration_sec=5.0)
            return
        
        # Get reference
        x_ref = self._get_current_reference()
        
        # Solve MPC
        result = self.mpc.solve(self.current_state, x_ref)
        
        # Publish status
        status_msg = String()
        status_msg.data = result['status']
        self.status_pub.publish(status_msg)
        
        # Publish command
        if 'optimal' in result['status'] or 'fallback' in result['status']:
            cmd_msg = self._build_accel_msg(result['u'])
            self.cmd_pub.publish(cmd_msg)
    
    def _build_accel_msg(self, u: np.ndarray) -> Accel:
        """Build Accel message from control input."""
        msg = Accel()
        msg.linear.x = u[0]
        msg.linear.y = u[1]
        msg.linear.z = u[2]
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = TubeMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
