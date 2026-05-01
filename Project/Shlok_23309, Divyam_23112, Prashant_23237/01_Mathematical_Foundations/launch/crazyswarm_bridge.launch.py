"""
Launch file for Crazyswarm Bridge Node.

This bridge converts MPC acceleration commands to Crazyswarm2 FullState messages.

Usage:
    ros2 launch mpc_controllers crazyswarm_bridge.launch.py
    
Optional arguments:
    drone_name:=cf1              # Drone name (determines publish topic)
    prediction_horizon:=1        # Prediction steps for position/velocity
    dt:=0.05                     # Control timestep (should match MPC)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    drone_name_arg = DeclareLaunchArgument(
        'drone_name',
        default_value='cf10',
        description='Drone name for Crazyswarm2 (topic: /{drone_name}/cmd_full_state)'
    )
    
    prediction_horizon_arg = DeclareLaunchArgument(
        'prediction_horizon',
        default_value='1',
        description='Prediction horizon steps for position/velocity lookahead'
    )
    
    dt_arg = DeclareLaunchArgument(
        'dt',
        default_value='0.05',
        description='Control timestep in seconds (should match MPC controller)'
    )
    
    # Crazyswarm Bridge Node
    bridge_node = Node(
        package='lcs',
        executable='crazyswarm_bridge_node',
        name='crazyswarm_bridge_node',
        output='screen',
        parameters=[{
            'drone_name': LaunchConfiguration('drone_name'),
            'prediction_horizon': LaunchConfiguration('prediction_horizon'),
            'dt': LaunchConfiguration('dt'),
        }],
        remappings=[
            ('/drone/cmd_accel', '/drone/cmd_accel'),
            ('/drone/odom', ['/', LaunchConfiguration('drone_name'), '/odom']),
        ]
    )
    
    return LaunchDescription([
        drone_name_arg,
        prediction_horizon_arg,
        dt_arg,
        bridge_node,
    ])
