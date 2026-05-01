#!/usr/bin/env python3
"""
Waypoint Publisher Node

Automatically publishes waypoints to the /waypoint topic for the MPC controllers.
Can be configured with a list of waypoints and publishes them sequentially.

Subscriptions:
    /drone/odom (nav_msgs/Odometry): Optional, for monitoring waypoint completion

Publications:
    /waypoint (geometry_msgs/Point): Waypoint positions

Parameters:
    waypoints: Comma-separated x,y,z coordinates (default: "0.5,1.0,1.5")
    publish_rate: Hz to check and publish next waypoint (default: 1.0)
    waypoint_delay: Seconds to wait after reaching waypoint before next (default: 2.0)
    loop: Whether to loop waypoints indefinitely (default: false)
    start_delay: Seconds to wait before starting (default: 3.0)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import numpy as np
from typing import List


class WaypointPublisherNode(Node):
    """ROS2 node to automatically publish waypoints."""
    
    def __init__(self):
        super().__init__('waypoint_publisher_node')
        
        # Parameters
        self.declare_parameter('waypoints', '0.5,1.0,1.5')
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('waypoint_delay', 2.0)
        self.declare_parameter('waypoint_threshold', 0.15)
        self.declare_parameter('loop', False)
        self.declare_parameter('start_delay', 3.0)
        
        # Get parameters
        waypoints_str = self.get_parameter('waypoints').value
        self.waypoints = self._parse_waypoints(waypoints_str)
        publish_rate = self.get_parameter('publish_rate').value
        self.waypoint_delay = self.get_parameter('waypoint_delay').value
        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        self.loop = self.get_parameter('loop').value
        start_delay = self.get_parameter('start_delay').value
        
        # State
        self.current_wp_idx = 0
        self.current_odom = None
        self.wp_reached_time = None
        self.waiting_for_delay = False
        self.start_time = self.get_clock().now()
        self.started = False
        
        # QoS
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publisher
        self.waypoint_pub = self.create_publisher(Point, '/waypoint', 10)
        
        # Subscriber (optional, for monitoring)
        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, qos)
        
        # Timer
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)
        
        # Log initialization
        self._log_initialization(start_delay)
    
    def _parse_waypoints(self, waypoints_str: str) -> List[np.ndarray]:
        """Parse comma-separated waypoints string into list of arrays."""
        try:
            # Parse comma-separated values
            values = [float(x.strip()) for x in waypoints_str.split(',')]
            
            # Group into sets of 3 (x, y, z)
            if len(values) % 3 != 0:
                self.get_logger().error(f'Waypoints string has {len(values)} values, expected multiple of 3')
                return [np.array([0.5, 1.0, 1.5])]
            
            waypoints = []
            for i in range(0, len(values), 3):
                waypoints.append(np.array([values[i], values[i+1], values[i+2]]))
            
            return waypoints
        except Exception as e:
            self.get_logger().error(f'Failed to parse waypoints: {e}')
            return [np.array([0.5, 1.0, 1.5])]


    def _log_initialization(self, start_delay: float):
        """Log initialization info."""
        self.get_logger().info('Waypoint Publisher initialized')
        self.get_logger().info(f'  Waypoints: {len(self.waypoints)}')
        for i, wp in enumerate(self.waypoints):
            self.get_logger().info(f'    [{i}] x={wp[0]:.2f}, y={wp[1]:.2f}, z={wp[2]:.2f}')
        self.get_logger().info(f'  Loop: {self.loop}')
        self.get_logger().info(f'  Starting in {start_delay:.1f} seconds...')
    
    def _odom_callback(self, msg: Odometry):
        """Store current odometry."""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.current_odom = np.array([px, py, pz])
    
    def _timer_callback(self):
        """Main timer callback - publish waypoints."""
        # Check start delay
        if not self.started:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed < self.get_parameter('start_delay').value:
                return
            self.started = True
            self.get_logger().info('Starting waypoint publication!')
            self._publish_current_waypoint()
            return
        
        # Check if we have waypoints left
        if self.current_wp_idx >= len(self.waypoints):
            if self.loop:
                self.current_wp_idx = 0
                self.get_logger().info('Looping back to first waypoint')
            else:
                return  # Done
        
        # Check if current waypoint is reached
        if self.current_odom is not None and self.wp_reached_time is None:
            target = self.waypoints[self.current_wp_idx]
            dist = np.linalg.norm(self.current_odom - target)
            
            if dist < self.waypoint_threshold:
                self.wp_reached_time = self.get_clock().now()
                self.waiting_for_delay = True
                self.get_logger().info(
                    f'Waypoint [{self.current_wp_idx}] reached! '
                    f'Waiting {self.waypoint_delay:.1f}s...'
                )
        
        # Check if delay has passed
        if self.waiting_for_delay:
            elapsed = (self.get_clock().now() - self.wp_reached_time).nanoseconds / 1e9
            if elapsed >= self.waypoint_delay:
                self.current_wp_idx += 1
                self.wp_reached_time = None
                self.waiting_for_delay = False
                
                if self.current_wp_idx < len(self.waypoints) or self.loop:
                    self._publish_current_waypoint()
                else:
                    self.get_logger().info('All waypoints completed!')
    
    def _publish_current_waypoint(self):
        """Publish the current waypoint."""
        if self.current_wp_idx >= len(self.waypoints):
            return
        
        wp = self.waypoints[self.current_wp_idx]
        msg = Point()
        msg.x = float(wp[0])
        msg.y = float(wp[1])
        msg.z = float(wp[2])
        
        self.waypoint_pub.publish(msg)
        self.get_logger().info(
            f'Published waypoint [{self.current_wp_idx}/{len(self.waypoints)-1}]: '
            f'x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
