#!/usr/bin/env python3
"""
Crazyswarm Bridge Node.

Bridges MPC acceleration commands to the FullState setpoint that
Crazyswarm2 forwards to cf2 firmware.

Architecture:

    et_tube_mpc_node                       crazyswarm_bridge_node
        |   /drone/cmd_accel  ──▶              |
        |   /drone/nominal_pose ──▶            ▼
        +────/cf_1/odom (current state)──▶ build FullState
                                               │
                                               ▼
                                  /cf_1/cmd_full_state ──▶ Crazyswarm2 ──▶ cf2 ──▶ Gazebo

Key change versus the original implementation: the position setpoint comes
from the MPC nominal trajectory (`/drone/nominal_pose`) — not from a tiny
prediction off the current odom. The tiny prediction was sending setpoints
on the order of millimetres above the current position, which cannot lift
the drone off the ground. The MPC nominal trajectory is the correct
reference: it climbs from initial state toward the next waypoint over the
horizon, so cmd_full_state.pose drives cf2 to the right place.

Velocity is propagated from current odom plus a one-step accel kick
(vel + acc * dt). Acceleration is the MPC command directly. Orientation is
aligned with thrust direction (accel + g).

Subscriptions:
    /drone/cmd_accel       geometry_msgs/Accel        (MPC accel command)
    /drone/odom            nav_msgs/Odometry          (current state)
    /drone/nominal_pose    geometry_msgs/PoseStamped  (MPC nominal pose)

Publications:
    /<drone_name>/cmd_full_state   crazyflie_interfaces/FullState
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Accel, PoseStamped
from nav_msgs.msg import Odometry
from crazyflie_interfaces.msg import FullState
import numpy as np


class CrazyswarmBridgeNode(Node):
    """Bridge between MPC acceleration commands and Crazyswarm2 FullState."""

    def __init__(self):
        super().__init__('crazyswarm_bridge_node')

        self.declare_parameter('drone_name', 'cf_1')
        self.declare_parameter('dt', 0.05)
        # If True, fall back to a current-state predict when nominal_pose
        # has not arrived yet (only useful for the first ~50 ms of the run).
        self.declare_parameter('fallback_predict', True)

        drone_name = self.get_parameter('drone_name').value
        self.dt = self.get_parameter('dt').value
        self.fallback_predict = self.get_parameter('fallback_predict').value

        self.current_odom = None
        self.latest_nominal = None  # np.array([x, y, z])

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Accel, '/drone/cmd_accel', self._accel_callback, 10)
        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, qos_be)
        self.create_subscription(PoseStamped, '/drone/nominal_pose', self._nominal_callback, 10)

        cmd_topic = f'/{drone_name}/cmd_full_state'
        self.cmd_pub = self.create_publisher(FullState, cmd_topic, 10)

        self.get_logger().info('Crazyswarm Bridge initialized')
        self.get_logger().info(f'  Subscribing to: /drone/cmd_accel, /drone/odom, /drone/nominal_pose')
        self.get_logger().info(f'  Publishing to: {cmd_topic}')

    def _odom_callback(self, msg: Odometry):
        self.current_odom = msg

    def _nominal_callback(self, msg: PoseStamped):
        self.latest_nominal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def _accel_callback(self, msg: Accel):
        if self.current_odom is None:
            self.get_logger().warn('No odometry received yet, skipping command',
                                   throttle_duration_sec=5.0)
            return

        px = self.current_odom.pose.pose.position.x
        py = self.current_odom.pose.pose.position.y
        pz = self.current_odom.pose.pose.position.z
        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        vz = self.current_odom.twist.twist.linear.z

        ax = msg.linear.x
        ay = msg.linear.y
        az = msg.linear.z

        # Position setpoint: MPC nominal pose if available, else fallback.
        if self.latest_nominal is not None:
            tx, ty, tz = float(self.latest_nominal[0]), float(self.latest_nominal[1]), float(self.latest_nominal[2])
        elif self.fallback_predict:
            tx = px + vx * self.dt + 0.5 * ax * self.dt ** 2
            ty = py + vy * self.dt + 0.5 * ay * self.dt ** 2
            tz = pz + vz * self.dt + 0.5 * az * self.dt ** 2
        else:
            self.get_logger().warn('No nominal_pose received yet, skipping command',
                                   throttle_duration_sec=5.0)
            return

        # Velocity setpoint: current vel + accel kick over dt.
        vx_s = vx + ax * self.dt
        vy_s = vy + ay * self.dt
        vz_s = vz + az * self.dt

        cmd_msg = FullState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'world'
        cmd_msg.pose.position.x = tx
        cmd_msg.pose.position.y = ty
        cmd_msg.pose.position.z = tz
        cmd_msg.twist.linear.x = vx_s
        cmd_msg.twist.linear.y = vy_s
        cmd_msg.twist.linear.z = vz_s
        cmd_msg.acc.x = ax
        cmd_msg.acc.y = ay
        cmd_msg.acc.z = az

        q = self._compute_orientation(np.array([ax, ay, az]))
        cmd_msg.pose.orientation.x = q[0]
        cmd_msg.pose.orientation.y = q[1]
        cmd_msg.pose.orientation.z = q[2]
        cmd_msg.pose.orientation.w = q[3]

        cmd_msg.twist.angular.x = 0.0
        cmd_msg.twist.angular.y = 0.0
        cmd_msg.twist.angular.z = 0.0

        self.cmd_pub.publish(cmd_msg)

    def _compute_orientation(self, acceleration: np.ndarray) -> list:
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
