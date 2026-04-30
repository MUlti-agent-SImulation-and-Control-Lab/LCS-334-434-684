import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
import math

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        self.create_subscription(PoseStamped, '/waypoints', self.waypoint_callback, 10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # Control Parameters
        self.max_linear_speed = 0.22
        self.stop_distance = 0.1
        
        self.get_logger().info('Pure Pursuit Node Started (Publishing TwistStamped)')

    def waypoint_callback(self, msg):
        # Waypoint is in Robot Frame: X=Forward, Y=Left
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        distance = math.sqrt(x*x + y*y)
        
        # Create TwistStamped
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        
        if distance > self.stop_distance:
            # Pure Pursuit Logic: Curvature = 2y / L^2
            if distance < 0.01:
                curvature = 0.0
            else:
                curvature = (2.0 * y) / (distance * distance)
            
            target_speed = self.max_linear_speed
            
            # Simple adaptive speed (slow down on sharp turns)
            if abs(curvature) > 1.0:
                target_speed *= 0.5
            
            angular_speed = target_speed * curvature
            
            # Clamp angular speed
            max_ang = 1.5
            angular_speed = max(min(angular_speed, max_ang), -max_ang)
            
            twist_msg.twist.linear.x = float(target_speed)
            twist_msg.twist.angular.z = float(angular_speed)
            
        else:
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.angular.z = 0.0
            
        self.publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PurePursuitNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()