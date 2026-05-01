#!/usr/bin/env python3
"""
Crazyswarm Bridge Node

Bridge between MPC controllers and Crazyswarm2.
Converts geometry_msgs/Accel to crazyflie_interfaces/FullState.

Architecture:
    MPC Controller Node (/drone/cmd_accel)
        |
        v
    Crazyswarm Bridge Node (this file)
        |
        v
    Crazyswarm2 (/cf1/cmd_full_state)
        |
        v
    Gazebo / Real Crazyflie

Subscriptions:
    /drone/cmd_accel (geometry_msgs/Accel): Acceleration commands from MPC
    /drone/odom (nav_msgs/Odometry): Current state for position/velocity

Publications:
    /cf1/cmd_full_state (crazyflie_interfaces/FullState): Full state command
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Accel
from nav_msgs.msg import Odometry
from crazyflie_interfaces.msg import FullState
import numpy as np


class CrazyswarmBridgeNode(Node):
    """Bridge between MPC acceleration commands and Crazyswarm2 FullState."""
    
    def __init__(self):
        super().__init__('crazyswarm_bridge_node')
        
        # Parameters
        self.declare_parameter('drone_name', 'cf1')
        self.declare_parameter('prediction_horizon', 1)  # Steps to predict ahead
        self.declare_parameter('dt', 0.05)  # Control timestep
        
        drone_name = self.get_parameter('drone_name').value
        self.prediction_horizon = self.get_parameter('prediction_horizon').value
        self.dt = self.get_parameter('dt').value
        
        # State storage
        self.current_odom = None
        self.latest_accel = None
        
        # QoS profile
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.create_subscription(Accel, '/drone/cmd_accel', self._accel_callback, 10)
        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, qos)
        
        # Publisher
        cmd_topic = f'/{drone_name}/cmd_full_state'
        self.cmd_pub = self.create_publisher(FullState, cmd_topic, 10)
        
        self.get_logger().info(f'Crazyswarm Bridge initialized')
        self.get_logger().info(f'  Subscribing to: /drone/cmd_accel, /drone/odom')
        self.get_logger().info(f'  Publishing to: {cmd_topic}')
    
    def _odom_callback(self, msg: Odometry):
        """Store current odometry."""
        self.current_odom = msg
    
    def _accel_callback(self, msg: Accel):
        """Convert acceleration command to FullState and publish."""
        if self.current_odom is None:
            self.get_logger().warn('No odometry received yet, skipping command', throttle_duration_sec=5.0)
            return
        
        # Extract current state from odometry
        px = self.current_odom.pose.pose.position.x
        py = self.current_odom.pose.pose.position.y
        pz = self.current_odom.pose.pose.position.z
        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        vz = self.current_odom.twist.twist.linear.z
        
        # MPC acceleration command
        ax = msg.linear.x
        ay = msg.linear.y
        az = msg.linear.z
        
        # Predict future position using double integrator model
        # x_pred = x + v*dt + 0.5*a*dt^2
        dt = self.dt * self.prediction_horizon
        px_pred = px + vx * dt + 0.5 * ax * dt**2
        py_pred = py + vy * dt + 0.5 * ay * dt**2
        pz_pred = pz + vz * dt + 0.5 * az * dt**2
        
        # Predict future velocity
        vx_pred = vx + ax * dt
        vy_pred = vy + ay * dt
        vz_pred = vz + az * dt
        
        # Build FullState message
        cmd_msg = FullState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'world'
        
        # Position (predicted)
        cmd_msg.pose.position.x = px_pred
        cmd_msg.pose.position.y = py_pred
        cmd_msg.pose.position.z = pz_pred
        
        # Velocity (predicted)
        cmd_msg.twist.linear.x = vx_pred
        cmd_msg.twist.linear.y = vy_pred
        cmd_msg.twist.linear.z = vz_pred
        
        # Acceleration (from MPC)
        cmd_msg.acc.x = ax
        cmd_msg.acc.y = ay
        cmd_msg.acc.z = az
        
        # Compute orientation from acceleration vector
        # Align thrust with desired acceleration
        q = self._compute_orientation(np.array([ax, ay, az]))
        cmd_msg.pose.orientation.x = q[0]
        cmd_msg.pose.orientation.y = q[1]
        cmd_msg.pose.orientation.z = q[2]
        cmd_msg.pose.orientation.w = q[3]
        
        # Angular rates (zero - attitude controller handles this)
        cmd_msg.twist.angular.x = 0.0
        cmd_msg.twist.angular.y = 0.0
        cmd_msg.twist.angular.z = 0.0
        
        # Publish
        self.cmd_pub.publish(cmd_msg)
    
    def _compute_orientation(self, acceleration: np.ndarray) -> list:
        """
        Compute quaternion to align thrust with acceleration.
        
        Args:
            acceleration: Desired acceleration [ax, ay, az]
            
        Returns:
            Quaternion [qx, qy, qz, qw]
        """
        g = np.array([0, 0, 9.81])
        a_des = acceleration + g
        a_norm = np.linalg.norm(a_des)
        
        if a_norm < 1e-6:
            return [0.0, 0.0, 0.0, 1.0]
        
        a_des = a_des / a_norm
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, a_des)
        v_norm = np.linalg.norm(v)
        
        if v_norm < 1e-6:
            if np.dot(z_axis, a_des) > 0:
                return [0.0, 0.0, 0.0, 1.0]
            else:
                return [1.0, 0.0, 0.0, 0.0]
        
        v = v / v_norm
        theta = np.arccos(np.clip(np.dot(z_axis, a_des), -1.0, 1.0))
        qw = np.cos(theta / 2)
        qxyz = v * np.sin(theta / 2)
        
        return [qxyz[0], qxyz[1], qxyz[2], qw]


def main(args=None):
    rclpy.init(args=args)
    node = CrazyswarmBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
