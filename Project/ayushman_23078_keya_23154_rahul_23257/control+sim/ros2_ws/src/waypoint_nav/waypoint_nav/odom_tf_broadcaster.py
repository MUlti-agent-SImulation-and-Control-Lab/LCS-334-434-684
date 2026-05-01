#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class OdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Odometry,
            '/a200_0000/platform/odom/filtered',  # your exact topic
            self.odom_callback,
            10
        )
        self.get_logger().info('Odom TF Broadcaster started...')

    def odom_callback(self, msg: Odometry):
        t = TransformStamped()

        # Use the current ROS time (preferred)
        t.header.stamp = msg.header.stamp

        t.header.frame_id = msg.header.frame_id      # e.g. "odom"
        t.child_frame_id  = msg.child_frame_id       # e.g. "base_link"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation      = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    rclpy.spin(OdomTFBroadcaster())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
