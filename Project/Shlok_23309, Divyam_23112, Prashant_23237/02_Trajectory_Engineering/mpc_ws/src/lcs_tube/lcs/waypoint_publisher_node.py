#!/usr/bin/env python3
"""
Waypoint Publisher Node.

Publishes a sequence of waypoints to /waypoint, advancing automatically when
the drone is within waypoint_threshold of the current target. Two ways to
specify the sequence:

  (a) trajectory parameter ('M' or 'S') + trajectories_path parameter
      pointing to a python file that defines a TRAJECTORIES dict (Phase 2
      default: /root/mpc_ws/phase2/trajectories.py).

  (b) waypoints parameter as comma-separated 'x,y,z,x,y,z,...' (legacy).

Subscriptions:
    /drone/odom (nav_msgs/Odometry): for waypoint-reached check.

Publications:
    /waypoint (geometry_msgs/Point): the active target.
"""

import importlib.util
import os
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
        self.declare_parameter('trajectory', '')
        self.declare_parameter('trajectories_path', '/root/mpc_ws/phase2/trajectories.py')
        self.declare_parameter('waypoints', '0.5,1.0,1.5')
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('waypoint_delay', 2.0)
        self.declare_parameter('waypoint_threshold', 0.15)
        self.declare_parameter('loop', False)
        self.declare_parameter('start_delay', 0.0)

        trajectory = self.get_parameter('trajectory').value
        trajectories_path = self.get_parameter('trajectories_path').value
        waypoints_str = self.get_parameter('waypoints').value

        self.waypoints = self._load_waypoints(trajectory, trajectories_path, waypoints_str)
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

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.waypoint_pub = self.create_publisher(Point, '/waypoint', 10)
        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, qos)

        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        self._log_initialization(start_delay, trajectory, trajectories_path)

    def _load_waypoints(self, trajectory: str, trajectories_path: str,
                        waypoints_str: str) -> List[np.ndarray]:
        """Resolve waypoints from either a named trajectory or the legacy string."""
        if trajectory:
            wps = self._load_named_trajectory(trajectory, trajectories_path)
            if wps is not None:
                return wps
            self.get_logger().warn(
                f"Could not load trajectory {trajectory!r} from {trajectories_path}; "
                f"falling back to waypoints string.")
        return self._parse_waypoints(waypoints_str)

    def _load_named_trajectory(self, name: str, path: str):
        if not path or not os.path.isfile(path):
            self.get_logger().error(f'trajectories_path not found: {path}')
            return None
        try:
            spec = importlib.util.spec_from_file_location("user_trajectories", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            self.get_logger().error(f'Failed to import trajectories from {path}: {e}')
            return None

        traj_dict = getattr(module, 'TRAJECTORIES', None)
        if traj_dict is None:
            self.get_logger().error(
                f'{path} does not define TRAJECTORIES dict')
            return None
        traj = traj_dict.get(name.upper())
        if traj is None:
            self.get_logger().error(
                f'TRAJECTORIES has no key {name!r}; available: {sorted(traj_dict.keys())}')
            return None
        try:
            return [np.array(p, dtype=float) for p in traj]
        except Exception as e:
            self.get_logger().error(f'Bad trajectory shape for {name!r}: {e}')
            return None

    def _parse_waypoints(self, waypoints_str: str) -> List[np.ndarray]:
        try:
            values = [float(x.strip()) for x in waypoints_str.split(',')]
            if len(values) % 3 != 0:
                self.get_logger().error(
                    f'Waypoints string has {len(values)} values, expected multiple of 3')
                return [np.array([0.5, 1.0, 1.5])]
            waypoints = []
            for i in range(0, len(values), 3):
                waypoints.append(np.array([values[i], values[i+1], values[i+2]]))
            return waypoints
        except Exception as e:
            self.get_logger().error(f'Failed to parse waypoints: {e}')
            return [np.array([0.5, 1.0, 1.5])]

    def _log_initialization(self, start_delay: float, trajectory: str, trajectories_path: str):
        self.get_logger().info('Waypoint Publisher initialized')
        if trajectory:
            self.get_logger().info(f'  Trajectory: {trajectory} (from {trajectories_path})')
        self.get_logger().info(f'  Waypoints: {len(self.waypoints)}')
        for i, wp in enumerate(self.waypoints):
            self.get_logger().info(f'    [{i}] x={wp[0]:.2f}, y={wp[1]:.2f}, z={wp[2]:.2f}')
        self.get_logger().info(f'  Threshold: {self.waypoint_threshold:.2f}m, Delay: {self.waypoint_delay:.1f}s, Loop: {self.loop}')
        self.get_logger().info(f'  Starting in {start_delay:.1f} seconds...')

    def _odom_callback(self, msg: Odometry):
        self.current_odom = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])

    def _timer_callback(self):
        if not self.started:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed < self.get_parameter('start_delay').value:
                return
            self.started = True
            self.get_logger().info('Starting waypoint publication!')
            self._publish_current_waypoint()
            return

        if self.current_wp_idx >= len(self.waypoints):
            if self.loop:
                self.current_wp_idx = 0
                self.get_logger().info('Looping back to first waypoint')
            else:
                return

        if self.current_odom is not None and self.wp_reached_time is None:
            target = self.waypoints[self.current_wp_idx]
            dist = np.linalg.norm(self.current_odom - target)
            if dist < self.waypoint_threshold:
                self.wp_reached_time = self.get_clock().now()
                self.waiting_for_delay = True
                self.get_logger().info(
                    f'Waypoint [{self.current_wp_idx}] reached! '
                    f'Waiting {self.waypoint_delay:.1f}s...')

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
            f'x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}')


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
