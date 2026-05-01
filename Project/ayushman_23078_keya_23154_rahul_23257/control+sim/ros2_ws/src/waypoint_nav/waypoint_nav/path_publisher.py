# waypoint_nav/waypoint_nav/path_publisher.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import csv
import os
from ament_index_python.packages import get_package_share_directory


class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')

        self.declare_parameter('csv_file', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 1.0) 
        frame_id    = self.get_parameter('frame_id').value
        publish_rate = self.get_parameter('publish_rate').value

        csv_file = self.get_parameter('csv_file').value
        if not csv_file:
            pkg = get_package_share_directory('waypoint_nav')
            csv_file = os.path.join(pkg, 'config', 'waypoints.csv')

        self.waypoints = self._load_csv(csv_file)
        self.frame_id  = frame_id

        self.path_pub   = self.create_publisher(Path,        '/planned_path',    10)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)

        self.timer = self.create_timer(1.0 / publish_rate, self.publish_all)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {csv_file}')

    def _load_csv(self, path):
        waypoints = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoints.append((float(row['x']), float(row['y']), 0.0))  
        return waypoints

    def _make_header(self):
        h = Header()
        h.stamp    = rclpy.time.Time().to_msg()
        h.frame_id = self.frame_id
        return h

    def publish_all(self):
        self._publish_path()
        self._publish_markers()

    def _publish_path(self):
        path      = Path()
        path.header = self._make_header()

        for x, y, z in self.waypoints:
            ps = PoseStamped()
            ps.header         = self._make_header()
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0  # identity quaternion
            path.poses.append(ps)

        self.path_pub.publish(path)

    def _publish_markers(self):
        marker_array = MarkerArray()

        for i, (x, y, z) in enumerate(self.waypoints):
            m = Marker()
            m.header      = self._make_header()
            m.ns          = 'waypoints'
            m.id          = i
            m.type        = Marker.SPHERE
            m.action      = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.4
            m.color.r = 0.0; m.color.g = 0.8; m.color.b = 0.2; m.color.a = 1.0
            marker_array.markers.append(m)

            t = Marker()
            t.header      = self._make_header()
            t.ns          = 'labels'
            t.id          = i + 1000
            t.type        = Marker.TEXT_VIEW_FACING
            t.action      = Marker.ADD
            t.pose.position.x = x
            t.pose.position.y = y
            t.pose.position.z = z + 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.3
            t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0; t.color.a = 1.0
            t.text = str(i)
            marker_array.markers.append(t)
            
        line = Marker()
        line.header  = self._make_header()
        line.ns      = 'path_line'
        line.id      = 9999
        line.type    = Marker.LINE_STRIP
        line.action  = Marker.ADD
        line.scale.x = 0.05
        line.color.r = 0.0; line.color.g = 0.4; line.color.b = 1.0; line.color.a = 0.8
        for x, y, z in self.waypoints:
            p = Point(); p.x = x; p.y = y; p.z = z
            line.points.append(p)
        marker_array.markers.append(line)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
