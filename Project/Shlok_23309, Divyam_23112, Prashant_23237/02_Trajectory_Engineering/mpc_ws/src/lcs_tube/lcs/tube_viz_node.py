#!/usr/bin/env python3
"""
Tube visualization node for ET-Tube MPC.

Renders mRPI tube + supporting markers in RViz2:
  - Translucent cylinder around the nominal pose, sized from the LP-projected
    bounding box of Omega_H/Omega_h on the position subspace.
  - Small spheres at current odom (cyan) and nominal pose (yellow).
  - Persistent green spheres at each received waypoint.
  - Transient red spheres at the current odom whenever an et_trigger fires.

Subscriptions:
    /drone/odom           nav_msgs/Odometry      (BEST_EFFORT VOLATILE)
    /drone/nominal_pose   geometry_msgs/PoseStamped
    /drone/et_trigger     std_msgs/String        (BEST_EFFORT VOLATILE)
    /waypoint             geometry_msgs/Point

Publications:
    /tube_viz             visualization_msgs/MarkerArray
"""

from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
)
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from scipy.optimize import linprog


def _qos_be(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


def _support(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    res = linprog(c=-c, A_ub=A, b_ub=b, bounds=[(-1e6, 1e6)] * A.shape[1], method='highs')
    if res.success:
        return float(-res.fun)
    return float('-inf')


def compute_position_bbox(Omega_H: np.ndarray, Omega_h: np.ndarray):
    n = Omega_H.shape[1]
    bounds = []
    for axis in range(3):
        c_pos = np.zeros(n); c_pos[axis] = 1.0
        c_neg = np.zeros(n); c_neg[axis] = -1.0
        hi = _support(c_pos, Omega_H, Omega_h)
        lo = -_support(c_neg, Omega_H, Omega_h)
        bounds.append((lo, hi))
    return bounds


# Persistent (lifetime=0 means never expire in RViz)
_PERSIST = Duration(sec=0, nanosec=0)


class TubeVizNode(Node):
    def __init__(self):
        super().__init__('tube_viz_node')

        base_dir = Path(__file__).resolve().parent.parent
        default_tube_path = str(base_dir / 'simulations' / 'tube_data.npz')

        self.declare_parameter('tube_data_path', default_tube_path)
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('trigger_lifetime_s', 2.0)
        self.declare_parameter('cylinder_height_clip', 0.5)

        tube_data_path = self.get_parameter('tube_data_path').value
        self.frame_id = self.get_parameter('frame_id').value
        publish_rate = self.get_parameter('publish_rate').value
        self.trigger_lifetime_s = self.get_parameter('trigger_lifetime_s').value
        cyl_h_clip = self.get_parameter('cylinder_height_clip').value

        data = np.load(tube_data_path)
        Omega_H = data['Omega_H']; Omega_h = data['Omega_h']
        bbox = compute_position_bbox(Omega_H, Omega_h)
        x_lo, x_hi = bbox[0]; y_lo, y_hi = bbox[1]; z_lo, z_hi = bbox[2]
        self.tube_radius = float(max(abs(x_lo), abs(x_hi), abs(y_lo), abs(y_hi)))
        raw_h = float(abs(z_hi - z_lo))
        self.tube_height = float(min(raw_h, cyl_h_clip)) if cyl_h_clip > 0 else raw_h
        self.get_logger().info(
            f"mRPI position bbox: x[{x_lo:+.3f}, {x_hi:+.3f}] "
            f"y[{y_lo:+.3f}, {y_hi:+.3f}] z[{z_lo:+.3f}, {z_hi:+.3f}]")
        self.get_logger().info(
            f"Tube cylinder: radius={self.tube_radius:.3f} m, height={self.tube_height:.3f} m")

        self.current_pose = None
        self.nominal_pose = None
        self.waypoints = []
        self.trigger_events = []  # (stamp_sec, x, y, z)

        self.create_subscription(Odometry, '/drone/odom', self._odom_cb, _qos_be(10))
        self.create_subscription(PoseStamped, '/drone/nominal_pose', self._nominal_cb, 10)
        self.create_subscription(String, '/drone/et_trigger', self._trigger_cb, _qos_be(10))
        self.create_subscription(Point, '/waypoint', self._waypoint_cb, 10)

        self.viz_pub = self.create_publisher(MarkerArray, '/tube_viz', 10)
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_markers)

        self.get_logger().info('Tube viz node initialized')

    def _odom_cb(self, msg: Odometry):
        self.current_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])

    def _nominal_cb(self, msg: PoseStamped):
        self.nominal_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def _trigger_cb(self, msg: String):
        if self.current_pose is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        self.trigger_events.append((now,
                                    float(self.current_pose[0]),
                                    float(self.current_pose[1]),
                                    float(self.current_pose[2])))
        cutoff = now - max(self.trigger_lifetime_s, 0.1) * 4.0
        self.trigger_events = [e for e in self.trigger_events if e[0] >= cutoff]

    def _waypoint_cb(self, msg: Point):
        self.waypoints.append(np.array([msg.x, msg.y, msg.z]))

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _make_sphere(self, marker_id: int, x: float, y: float, z: float,
                     scale: float, color: ColorRGBA, lifetime_s: float = 0.0) -> Marker:
        m = Marker()
        m.header.stamp = self._stamp()
        m.header.frame_id = self.frame_id
        m.ns = 'tube_viz'
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(scale); m.scale.y = float(scale); m.scale.z = float(scale)
        m.color = color
        if lifetime_s > 0.0:
            m.lifetime = Duration(sec=int(lifetime_s), nanosec=int((lifetime_s % 1) * 1e9))
        else:
            m.lifetime = _PERSIST
        return m

    def _make_cylinder(self, marker_id: int, x: float, y: float, z: float,
                       radius: float, height: float, color: ColorRGBA) -> Marker:
        m = Marker()
        m.header.stamp = self._stamp()
        m.header.frame_id = self.frame_id
        m.ns = 'tube_viz'
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(2.0 * radius)
        m.scale.y = float(2.0 * radius)
        m.scale.z = float(height)
        m.color = color
        m.lifetime = _PERSIST
        return m

    def _publish_markers(self):
        msg = MarkerArray()

        if self.nominal_pose is not None:
            tube_color = ColorRGBA(r=0.3, g=0.6, b=1.0, a=0.25)
            msg.markers.append(self._make_cylinder(
                0,
                self.nominal_pose[0], self.nominal_pose[1], self.nominal_pose[2],
                self.tube_radius, self.tube_height, tube_color))
            msg.markers.append(self._make_sphere(
                2,
                self.nominal_pose[0], self.nominal_pose[1], self.nominal_pose[2],
                0.06, ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)))

        if self.current_pose is not None:
            msg.markers.append(self._make_sphere(
                1,
                self.current_pose[0], self.current_pose[1], self.current_pose[2],
                0.05, ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)))

        for i, wp in enumerate(self.waypoints):
            msg.markers.append(self._make_sphere(
                100 + i,
                wp[0], wp[1], wp[2],
                0.10, ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.8)))

        now = self.get_clock().now().nanoseconds / 1e9
        keep = [e for e in self.trigger_events if (now - e[0]) <= self.trigger_lifetime_s]
        for i, (stamp, x, y, z) in enumerate(keep):
            msg.markers.append(self._make_sphere(
                200 + i,
                x, y, z,
                0.08, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9),
                lifetime_s=self.trigger_lifetime_s))
        # delete stale ones
        if len(self.trigger_events) > len(keep):
            stale = len(self.trigger_events) - len(keep)
            for i in range(len(keep), len(keep) + stale):
                d = Marker()
                d.header.stamp = self._stamp()
                d.header.frame_id = self.frame_id
                d.ns = 'tube_viz'
                d.id = 200 + i
                d.action = Marker.DELETE
                msg.markers.append(d)
        self.trigger_events = keep

        if msg.markers:
            self.viz_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TubeVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
