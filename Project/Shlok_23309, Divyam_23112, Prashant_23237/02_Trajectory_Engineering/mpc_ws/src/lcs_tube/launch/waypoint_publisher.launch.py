"""
Launch file for Waypoint Publisher Node.

Automatically publishes waypoints for MPC controllers to follow.

Usage:
    ros2 launch mpc_controllers waypoint_publisher.launch.py
    
Examples:
    # Single waypoint (x,y,z)
    ros2 launch mpc_controllers waypoint_publisher.launch.py waypoints:="0.5,1.0,1.5"
    
    # Multiple waypoints (x1,y1,z1,x2,y2,z2,...)
    ros2 launch mpc_controllers waypoint_publisher.launch.py waypoints:="0.5,0.0,1.0,0.5,1.0,1.5,0.0,1.0,1.0"
    
    # Loop waypoints indefinitely
    ros2 launch mpc_controllers waypoint_publisher.launch.py waypoints:="0.5,1.0,1.5" loop:=true
    
    # Faster waypoint progression
    ros2 launch mpc_controllers waypoint_publisher.launch.py waypoint_delay:=1.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    waypoints_arg = DeclareLaunchArgument(
        'waypoints',
        default_value='0.5,1.0,1.5',
        description='Comma-separated list of x,y,z coordinates (e.g., "x1,y1,z1,x2,y2,z2")'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='1.0',
        description='Rate to check waypoint progress (Hz)'
    )
    
    waypoint_delay_arg = DeclareLaunchArgument(
        'waypoint_delay',
        default_value='2.0',
        description='Seconds to wait at each waypoint before next'
    )
    
    waypoint_threshold_arg = DeclareLaunchArgument(
        'waypoint_threshold',
        default_value='0.15',
        description='Distance threshold to consider waypoint reached (m)'
    )
    
    loop_arg = DeclareLaunchArgument(
        'loop',
        default_value='false',
        description='Loop waypoints indefinitely'
    )
    
    start_delay_arg = DeclareLaunchArgument(
        'start_delay',
        default_value='3.0',
        description='Seconds to wait before starting'
    )
    
    drone_name_arg = DeclareLaunchArgument(
        'drone_name',
        default_value='cf10',
        description='Drone name (used for odometry topic)'
    )
    
    # Waypoint Publisher Node
    waypoint_publisher_node = Node(
        package='lcs',
        executable='waypoint_publisher_node',
        name='waypoint_publisher_node',
        output='screen',
        parameters=[{
            'waypoints': LaunchConfiguration('waypoints'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'waypoint_delay': LaunchConfiguration('waypoint_delay'),
            'waypoint_threshold': LaunchConfiguration('waypoint_threshold'),
            'loop': LaunchConfiguration('loop'),
            'start_delay': LaunchConfiguration('start_delay'),
        }],
        remappings=[
            ('/drone/odom', ['/', LaunchConfiguration('drone_name'), '/odom']),
            ('/waypoint', '/waypoint'),
        ]
    )
    
    return LaunchDescription([
        waypoints_arg,
        publish_rate_arg,
        waypoint_delay_arg,
        waypoint_threshold_arg,
        loop_arg,
        start_delay_arg,
        drone_name_arg,
        waypoint_publisher_node,
    ])
