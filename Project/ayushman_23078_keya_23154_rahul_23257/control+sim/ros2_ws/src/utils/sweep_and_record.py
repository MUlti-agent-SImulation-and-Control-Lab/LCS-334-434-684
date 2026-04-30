# twist_relay.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class TwistRelay(Node):
    def __init__(self):
        super().__init__('twist_relay')
        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cb, 10)
        self.pub = self.create_publisher(
            TwistStamped, '/a200_0000/platform/cmd_vel', 10)
        self.get_logger().info('Twist → TwistStamped relay running...')

    def cb(self, msg):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.twist = msg
        self.pub.publish(stamped)

def main():
    rclpy.init()
    node = TwistRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()