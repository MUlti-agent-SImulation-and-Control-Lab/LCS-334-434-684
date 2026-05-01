#!/usr/bin/env python3
"""
CFLib Bridge Node — direct CFLib connection to cf2 firmware.

On startup: opens CFLib (after a warm-up connect-disconnect cycle that cf2
SITL needs before its log block streaming becomes reliable), resets Kalman,
takes off via high_level_commander, starts a 50 Hz log block for state
estimate, and starts forwarding MPC commands as full-state setpoints.

Setpoint built from MPC's nominal trajectory (one step ahead):
  pos setpoint = /drone/nominal_state.pose      (x_traj[1] position)
  vel setpoint = /drone/nominal_state.twist     (x_traj[1] velocity)
  acc setpoint = /drone/cmd_accel               (MPC's u[0], feed-forward)

Subscriptions:
    /drone/cmd_accel       geometry_msgs/Accel        (BEST_EFFORT)
    /drone/nominal_state   nav_msgs/Odometry          (BEST_EFFORT)

Publications:
    /<drone_name>/odom     nav_msgs/Odometry          (BEST_EFFORT)
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
)

from geometry_msgs.msg import Accel
from nav_msgs.msg import Odometry

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig


def _qos_be(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


class CFLibBridgeNode(Node):
    def __init__(self):
        super().__init__('cflib_bridge_node')

        self.declare_parameter('uri', 'udp://127.0.0.1:19850')
        self.declare_parameter('drone_name', 'cf_1')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('child_frame_id', 'cf_1')
        self.declare_parameter('takeoff_height', 1.0)
        self.declare_parameter('takeoff_duration', 2.5)
        self.declare_parameter('estimator_settle', 2.0)
        self.declare_parameter('odom_period_ms', 20)
        self.declare_parameter('warmup_enabled', True)
        self.declare_parameter('warmup_hold_s', 0.3)
        self.declare_parameter('warmup_gap_s', 1.5)

        self._uri = self.get_parameter('uri').value
        self._drone_name = self.get_parameter('drone_name').value
        self._frame_id = self.get_parameter('frame_id').value
        self._child_frame_id = self.get_parameter('child_frame_id').value
        self._takeoff_h = self.get_parameter('takeoff_height').value
        self._takeoff_d = self.get_parameter('takeoff_duration').value
        self._settle = self.get_parameter('estimator_settle').value
        self._odom_period_ms = int(self.get_parameter('odom_period_ms').value)
        warmup_enabled = self.get_parameter('warmup_enabled').value
        warmup_hold = self.get_parameter('warmup_hold_s').value
        warmup_gap = self.get_parameter('warmup_gap_s').value

        self._lock = threading.Lock()
        self._latest_nominal = None
        self._is_armed = False

        odom_topic = f'/{self._drone_name}/odom'
        self._odom_pub = self.create_publisher(Odometry, odom_topic, _qos_be(10))
        self._cmd_sub = self.create_subscription(
            Accel, '/drone/cmd_accel', self._cmd_accel_cb, _qos_be(10))
        self._nominal_sub = self.create_subscription(
            Odometry, '/drone/nominal_state', self._nominal_state_cb, _qos_be(10))
        self.get_logger().info(
            f'odom topic: {odom_topic}; cmd_accel: /drone/cmd_accel; nominal_state: /drone/nominal_state')

        cflib.crtp.init_drivers()

        # --- Warm-up cycle ---
        # cf2 SITL's log block streaming is sometimes not enabled on the first
        # CFLib connect after a fresh cf2 process. A throwaway connect-disconnect
        # primes its plugin and makes the next connect's log block stream reliably.
        # Symptom without this: /cf_1/odom subscribers see 0 Hz, MPC logs
        # "No odometry received yet" forever even though the bridge has logged
        # "Odom log block started".
        if warmup_enabled:
            self.get_logger().info(
                f'CFLib warm-up cycle to {self._uri} (hold {warmup_hold}s, gap {warmup_gap}s)')
            try:
                warm = SyncCrazyflie(self._uri, cf=Crazyflie(rw_cache='/tmp/cf_cache'))
                warm.open_link()
                time.sleep(warmup_hold)
                warm.close_link()
                time.sleep(warmup_gap)
                self.get_logger().info('warm-up complete')
            except Exception as e:
                self.get_logger().warn(f'warm-up failed (continuing): {e}')

        self._scf = SyncCrazyflie(self._uri, cf=Crazyflie(rw_cache='/tmp/cf_cache'))
        self.get_logger().info(f'Opening CFLib link to {self._uri}')
        self._scf.open_link()
        self._cf = self._scf.cf
        self.get_logger().info('CFLib link is open')

        # Param TOC sometimes isn't ready yet under ros2 launch — wait briefly.
        time.sleep(1.0)

        self.get_logger().info('Kalman reset')
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(self._settle)

        self._cf.param.set_value('commander.enHighLevel', '1')
        time.sleep(0.1)

        self._start_log_block(self._odom_period_ms)
        time.sleep(0.3)

        self.get_logger().info(
            f'Takeoff to {self._takeoff_h:.2f} m over {self._takeoff_d:.2f} s')
        self._cf.high_level_commander.takeoff(self._takeoff_h, self._takeoff_d)
        time.sleep(self._takeoff_d + 0.5)

        with self._lock:
            self._is_armed = True

        print('=' * 60, flush=True)
        print('CFLIB BRIDGE READY - starting MPC', flush=True)
        print('=' * 60, flush=True)

    def _start_log_block(self, period_ms: int):
        log_conf = LogConfig(name='odom', period_in_ms=period_ms)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stateEstimate.vx', 'float')
        log_conf.add_variable('stateEstimate.vy', 'float')
        log_conf.add_variable('stateEstimate.vz', 'float')
        self._cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self._log_odom_cb)
        log_conf.error_cb.add_callback(
            lambda lc, m: self.get_logger().error(f'log error on {lc.name}: {m}'))
        log_conf.start()
        self._log_conf = log_conf
        self.get_logger().info(f'Odom log block started at {period_ms} ms')

    def _log_odom_cb(self, timestamp, data, logconf):
        x = float(data['stateEstimate.x'])
        y = float(data['stateEstimate.y'])
        z = float(data['stateEstimate.z'])
        vx = float(data['stateEstimate.vx'])
        vy = float(data['stateEstimate.vy'])
        vz = float(data['stateEstimate.vz'])
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.child_frame_id = self._child_frame_id
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.linear.z = vz
        self._odom_pub.publish(msg)

    def _nominal_state_cb(self, msg: Odometry):
        with self._lock:
            self._latest_nominal = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ])

    def _cmd_accel_cb(self, msg: Accel):
        with self._lock:
            nominal = None if self._latest_nominal is None else self._latest_nominal.copy()
            armed = self._is_armed
        if not armed or nominal is None:
            return

        ax = float(msg.linear.x)
        ay = float(msg.linear.y)
        az = float(msg.linear.z)
        try:
            self._cf.commander.send_full_state_setpoint(
                (float(nominal[0]), float(nominal[1]), float(nominal[2])),
                (float(nominal[3]), float(nominal[4]), float(nominal[5])),
                (ax, ay, az),
                (0.0, 0.0, 0.0, 1.0),
                0.0, 0.0, 0.0,
            )
        except Exception as e:
            self.get_logger().warn(
                f'send_full_state_setpoint failed: {e}',
                throttle_duration_sec=2.0,
            )

    def shutdown(self):
        with self._lock:
            self._is_armed = False
        try:
            self.get_logger().info('Landing')
            self._cf.high_level_commander.land(0.0, 2.5)
            time.sleep(3.0)
        except Exception:
            pass
        try:
            self._log_conf.stop()
        except Exception:
            pass
        try:
            self._scf.close_link()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CFLibBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
