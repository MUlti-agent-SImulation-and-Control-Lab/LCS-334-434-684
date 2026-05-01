#!/usr/bin/env python3
"""
Event-Triggered Tube MPC Controller Node.

Subscriptions:
    /drone/odom (nav_msgs/Odometry, BEST_EFFORT VOLATILE depth=10)
    /waypoint   (geometry_msgs/Point)

Publications:
    /drone/cmd_accel     (geometry_msgs/Accel,        BEST_EFFORT VOLATILE depth=10)
    /drone/mpc_status    (std_msgs/String,            BEST_EFFORT VOLATILE depth=10)
    /drone/et_trigger    (std_msgs/String,            BEST_EFFORT VOLATILE depth=10)
    /drone/nominal_pose  (geometry_msgs/PoseStamped)  x_traj[1] position only (for viz)
    /drone/nominal_state (nav_msgs/Odometry)          x_traj[1] pos + vel
        consumed by cflib_bridge to build cmd_full_state setpoint
"""

from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
)
from geometry_msgs.msg import Point, Accel, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
from typing import List

from lcs.mpc_solvers import ETTubeMPC


def _qos_be(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


class ETTubeMPCNode(Node):
    def __init__(self):
        super().__init__('et_tube_mpc_node')

        base_dir = Path(__file__).resolve().parent.parent
        default_tube_path = str(base_dir / 'simulations' / 'tube_data.npz')

        self.declare_parameter('horizon', 20)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('tube_data_path', default_tube_path)
        self.declare_parameter('waypoint_threshold', 0.15)
        self.declare_parameter('trigger_threshold', 0.15)
        self.declare_parameter('consecutive_threshold', 3)
        self.declare_parameter('frame_id', 'world')
        # Lookahead step in x_traj used for nominal_state (1 = next step, 50 ms ahead at dt=0.05)
        self.declare_parameter('nominal_lookahead', 1)

        horizon = self.get_parameter('horizon').value
        dt = self.get_parameter('dt').value
        control_rate = self.get_parameter('control_rate').value
        tube_data_path = self.get_parameter('tube_data_path').value
        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        trigger_threshold = self.get_parameter('trigger_threshold').value
        consecutive_threshold = self.get_parameter('consecutive_threshold').value
        self.frame_id = self.get_parameter('frame_id').value
        self.nominal_lookahead = int(self.get_parameter('nominal_lookahead').value)

        self.get_logger().info(f'Initializing ET-Tube MPC: horizon={horizon}, dt={dt}')
        self.get_logger().info(
            f'  Trigger threshold: {trigger_threshold}m, Consecutive: {consecutive_threshold}, '
            f'Waypoint threshold: {self.waypoint_threshold}m, Nominal lookahead: {self.nominal_lookahead}')

        self.mpc = ETTubeMPC(
            horizon=horizon,
            dt=dt,
            tube_data_path=tube_data_path,
            trigger_threshold=trigger_threshold,
            consecutive_threshold=consecutive_threshold,
        )
        self.mpc.reset()

        self.current_state = None
        self.waypoints: List[np.ndarray] = []
        self.current_wp_idx = 0
        self.default_hover = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        self.trigger_count = 0
        self.total_steps = 0

        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, _qos_be(10))
        self.create_subscription(Point, '/waypoint', self._waypoint_callback, 10)

        self.cmd_pub = self.create_publisher(Accel, '/drone/cmd_accel', _qos_be(10))
        self.status_pub = self.create_publisher(String, '/drone/mpc_status', _qos_be(10))
        self.trigger_pub = self.create_publisher(String, '/drone/et_trigger', _qos_be(10))
        self.nominal_pose_pub = self.create_publisher(PoseStamped, '/drone/nominal_pose', 10)
        self.nominal_state_pub = self.create_publisher(Odometry, '/drone/nominal_state', _qos_be(10))

        timer_period = 1.0 / control_rate
        self.timer = self.create_timer(timer_period, self._control_loop)
        self.get_logger().info('ET-Tube MPC Node initialized')

    def _odom_callback(self, msg: Odometry):
        self.current_state = np.array([
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
        ])

    def _waypoint_callback(self, msg: Point):
        waypoint = np.array([msg.x, msg.y, msg.z, 0.0, 0.0, 0.0])
        self.waypoints.append(waypoint)
        self.get_logger().info(
            f'Added waypoint: [{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}] '
            f'(Queue: {len(self.waypoints)})')

    def _get_current_reference(self) -> np.ndarray:
        if not self.waypoints:
            return self.default_hover
        if self.current_state is not None and self.current_wp_idx < len(self.waypoints):
            current_wp = self.waypoints[self.current_wp_idx]
            dist = np.linalg.norm(self.current_state[0:3] - current_wp[0:3])
            if dist < self.waypoint_threshold:
                self.current_wp_idx += 1
                if self.current_wp_idx < len(self.waypoints):
                    self.get_logger().info(
                        f'Waypoint reached. Moving to next ({self.current_wp_idx + 1}/{len(self.waypoints)})')
                else:
                    self.get_logger().info('All waypoints completed. Hovering at final position.')
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        elif self.waypoints:
            return self.waypoints[-1]
        else:
            return self.default_hover

    def _publish_nominal(self, x_traj):
        if x_traj is None:
            return
        try:
            arr = np.asarray(x_traj)
        except Exception:
            return
        if arr.ndim < 2 or arr.shape[1] < 6:
            return
        idx = min(self.nominal_lookahead, arr.shape[0] - 1)
        x_bar = arr[idx]

        # PoseStamped (for tube_viz)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x_bar[0])
        msg.pose.position.y = float(x_bar[1])
        msg.pose.position.z = float(x_bar[2])
        msg.pose.orientation.w = 1.0
        self.nominal_pose_pub.publish(msg)

        # Odometry with pos AND vel (for cflib_bridge)
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = 'cf_1'
        odom.pose.pose.position.x = float(x_bar[0])
        odom.pose.pose.position.y = float(x_bar[1])
        odom.pose.pose.position.z = float(x_bar[2])
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = float(x_bar[3])
        odom.twist.twist.linear.y = float(x_bar[4])
        odom.twist.twist.linear.z = float(x_bar[5])
        self.nominal_state_pub.publish(odom)

    def _control_loop(self):
        if self.current_state is None:
            self.get_logger().warn('No odometry received yet', throttle_duration_sec=5.0)
            return
        self.total_steps += 1
        x_ref = self._get_current_reference()
        result = self.mpc.solve(self.current_state, x_ref)

        status_msg = String()
        status_msg.data = result['status']
        self.status_pub.publish(status_msg)

        if result.get('trigger'):
            self.trigger_count += 1
            pos_error = result.get('pos_error', 0.0)
            trigger_msg = String()
            trigger_msg.data = (f"TRIGGER step={self.total_steps} error={pos_error:.3f}m "
                               f"triggers={self.trigger_count}")
            self.trigger_pub.publish(trigger_msg)
            if self.trigger_count % 10 == 0:
                rate = 100.0 * self.trigger_count / self.total_steps
                self.get_logger().info(
                    f'Event trigger #{self.trigger_count} at step {self.total_steps} '
                    f'({rate:.1f}% solve rate)')

        self._publish_nominal(result.get('x_traj'))

        if 'optimal' in result['status'] or 'fallback' in result['status'] or 'nominal_propagated' in result['status']:
            cmd_msg = self._build_accel_msg(result['u'])
            self.cmd_pub.publish(cmd_msg)

    def _build_accel_msg(self, u: np.ndarray) -> Accel:
        msg = Accel()
        msg.linear.x = float(u[0])
        msg.linear.y = float(u[1])
        msg.linear.z = float(u[2])
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        return msg

    def destroy_node(self):
        if self.total_steps > 0:
            rate = 100.0 * self.trigger_count / self.total_steps
            self.get_logger().info(
                f'Final statistics: {self.trigger_count}/{self.total_steps} '
                f'triggers ({rate:.1f}% solve rate)')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ETTubeMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
