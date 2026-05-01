import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped  # Changed Twist to TwistStamped
import math

def euler_from_quaternion(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return 0, 0, yaw

class StanleyNode(Node):
    def __init__(self):
        super().__init__('stanley_node')
        self.create_subscription(PoseStamped, '/waypoints', self.waypoint_callback, 10)
        # Change Publisher to TwistStamped
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.k_cte = 2.0
        self.max_linear_speed = 0.22
        self.stop_distance = 0.1
        self.get_logger().info('Stanley Node Started (Publishing TwistStamped)')

    def waypoint_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, target_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        dist = math.sqrt(x*x + y*y)
        
        # Create TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        
        if dist > self.stop_distance:
            psi_e = target_yaw
            psi_e = (psi_e + math.pi) % (2 * math.pi) - math.pi
            
            cte = y * math.cos(target_yaw) - x * math.sin(target_yaw)
            v = self.max_linear_speed
            
            delta = psi_e + math.atan2(self.k_cte * cte, 1.0 + v)
            delta = max(min(delta, 1.0), -1.0)
            
            angular_speed = v * math.tan(delta) / 0.2
            angular_speed = max(min(angular_speed, 1.5), -1.5)
            
            twist_msg.twist.linear.x = v
            twist_msg.twist.angular.z = angular_speed
            
        self.publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StanleyNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()