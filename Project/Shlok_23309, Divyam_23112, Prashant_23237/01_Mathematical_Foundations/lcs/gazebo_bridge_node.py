#!/usr/bin/env python3
"""
Gazebo Bridge Node

Translates MPC acceleration commands into Wrench (force) commands
for the simple generic Gazebo drone (using gazebo_ros_force).

Subscriptions:
    /drone/cmd_accel (geometry_msgs/Accel): Commanded acceleration
    
Publications:
    /drone/cmd_force (geometry_msgs/Wrench): Force to apply to drone
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Accel, Wrench

class GazeboBridgeNode(Node):
    def __init__(self):
        super().__init__('gazebo_bridge_node')
        
        # We assume the simple_drone.urdf has mass = 1.0 kg
        self.declare_parameter('mass', 1.0)
        self.mass = self.get_parameter('mass').value
        self.gravity = 9.81
        
        self.create_subscription(Accel, '/drone/cmd_accel', self._accel_callback, 10)
        self.force_pub = self.create_publisher(Wrench, '/drone/cmd_force', 10)
        
        self.get_logger().info(f'Gazebo Bridge started (Mass: {self.mass}kg)')
        
    def _accel_callback(self, msg: Accel):
        """Convert Accel to Wrench and add gravity compensation."""
        wrench_msg = Wrench()
        
        # F = m * a
        # Z-axis gets gravity compensation because the MPC outputs are assumed to be
        # relative to hover (i.e. az = 0 means hovering, az > 0 means climbing).
        wrench_msg.force.x = self.mass * msg.linear.x
        wrench_msg.force.y = self.mass * msg.linear.y
        wrench_msg.force.z = self.mass * (msg.linear.z + self.gravity)
        
        # No torques needed for this simplified point-mass-like control
        wrench_msg.torque.x = 0.0
        wrench_msg.torque.y = 0.0
        wrench_msg.torque.z = 0.0
        
        self.force_pub.publish(wrench_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GazeboBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
