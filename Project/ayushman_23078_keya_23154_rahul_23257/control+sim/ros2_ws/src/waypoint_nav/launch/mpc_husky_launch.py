"""
launch/mpc_husky.launch.py

Launches the MS-DeepONet MPC controller for the Husky A200 in Gazebo.

Usage (after building your package):
  ros2 launch <your_pkg> mpc_husky.launch.py model_path:=/abs/path/to/deepONet_model.pt

Optional args:
  waypoints_file      : /abs/path/to/waypoints.npy   (N,2 numpy array)
  dist_threshold      : 0.5
  use_gain_estimator  : true
  fault_enabled       : false
  fault_step          : 75
  fault_side          : left
  fault_magnitude     : 0.6
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Declare all overridable arguments ────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'model_path',
            default_value='/root/ros2_ws/src/waypoint_nav/config/deepONet_model.pt.zip',
            description='Absolute path to the trained DeepONet .pt file'
        ),
        DeclareLaunchArgument(
            'waypoints_file',
            default_value='/root/ros2_ws/src/waypoint_nav/config/waypoints.csv',
            description='Absolute path to .npy waypoints array (shape N x 2). '
                        'Leave empty to receive waypoints via /mpc/waypoints topic.'
        ),
        DeclareLaunchArgument(
            'dist_threshold',
            default_value='0.75',
            description='Distance (m) at which the robot switches to the next waypoint'
        ),
        DeclareLaunchArgument(
            'use_gain_estimator',
            default_value='true',
            description='Enable online wheel-efficiency estimation for fault tolerance'
        ),
        DeclareLaunchArgument(
            'fault_enabled',
            default_value='false',
            description='Inject a simulated wheel fault during the run'
        ),
        DeclareLaunchArgument(
            'fault_step',
            default_value='75',
            description='Control step at which the fault is injected'
        ),
        DeclareLaunchArgument(
            'fault_side',
            default_value='left',
            description="Which wheel to fault: 'left' or 'right'"
        ),
        DeclareLaunchArgument(
            'fault_magnitude',
            default_value='0.6',
            description='Wheel efficiency after fault (1.0 = healthy, 0.0 = dead)'
        ),
    ]

    mpc_node = Node(
        package='waypoint_nav',              
        executable='mpc_controller_node',   
        name='mpc_controller',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_path':         LaunchConfiguration('model_path'),
            'waypoints_file':     LaunchConfiguration('waypoints_file'),
            'dist_threshold':     LaunchConfiguration('dist_threshold'),
            'use_gain_estimator': LaunchConfiguration('use_gain_estimator'),
            'fault_enabled':      LaunchConfiguration('fault_enabled'),
            'fault_step':         LaunchConfiguration('fault_step'),
            'fault_side':         LaunchConfiguration('fault_side'),
            'fault_magnitude':    LaunchConfiguration('fault_magnitude'),
        }],
        remappings=[
            ('/odometry/filtered', '/a200_0000/platform/odometry/filtered'),
            ('/cmd_vel',           '/a200_0000/platform/cmd_vel')
        ]
    )

    return LaunchDescription(args + [
        LogInfo(msg='Starting MS-DeepONet MPC Controller for Husky A200'),
        mpc_node,
    ])