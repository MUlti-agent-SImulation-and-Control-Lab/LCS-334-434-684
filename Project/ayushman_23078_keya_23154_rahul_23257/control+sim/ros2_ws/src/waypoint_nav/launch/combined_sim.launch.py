from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    clearpath_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('clearpath_gz'),
                'launch',
                'simulation.launch.py'
            ])
        ),
        launch_arguments={
            'setup_path': '/root/ros2_ws/src/',
            'world': 'pipeline'
        }.items()
    )

    waypoint_visualize = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('waypoint_nav'),
                'launch',
                'visualize.launch.py'
            ])
        ),
        launch_arguments={
            'csv_file': LaunchConfiguration('waypoints_file'),
        }.items()
    )

    mpc_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('waypoint_nav'),
                'launch',
                'mpc_husky_launch.py'
            ])
        )
    )

    delayed_mpc = TimerAction(
        period=5.0,
        actions=[mpc_controller]
    )

    return LaunchDescription([
        clearpath_sim,
        waypoint_visualize,
        delayed_mpc
    ])
