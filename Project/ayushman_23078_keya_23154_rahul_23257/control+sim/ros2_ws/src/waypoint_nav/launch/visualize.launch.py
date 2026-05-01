from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_waypoint = get_package_share_directory('waypoint_nav')

    csv_default = os.path.join(pkg_waypoint, 'config', 'waypoints.csv')
    use_sim_time = True

    ekf_local_params = {
        'use_sim_time': use_sim_time,
        'frequency': 30.0,
        'sensor_timeout': 0.1,
        'two_d_mode': True,
        'publish_tf': True,
        'map_frame': 'map',
        'odom_frame': 'odom',
        'base_link_frame': 'base_link',
        'world_frame': 'odom',
        # Wheel Odometry from Husky
        'odom0': '/a200_0000/platform/odom',
        'odom0_config': [False, False, False, # X, Y, Z
                         False, False, False, # Roll, Pitch, Yaw
                         True,  True,  False, # VX, VY, VZ (Velocity is better for slip)
                         False, False, True,  # Vroll, Vpitch, Vyaw
                         False, False, False],
        # IMU from Husky
        'imu0': '/a200_0000/sensors/imu_0/data',
        'imu0_config': [False, False, False,
                        True,  True,  True,   # Roll, Pitch, Yaw
                        False, False, False,
                        True,  True,  True,   # Angular Velocity
                        True,  True,  True],  # Linear Acceleration
        'imu0_remove_gravitational_acceleration': True
    }

    ekf_global_params = ekf_local_params.copy()
    ekf_global_params['world_frame'] = 'map'
    ekf_global_params['odom1'] = '/odometry/gps'
    ekf_global_params['odom1_config'] = [True,  True,  False, # X, Y (from GPS)
                                         False, False, False,
                                         False, False, False,
                                         False, False, False,
                                         False, False, False]

    return LaunchDescription([
        DeclareLaunchArgument('csv_file', default_value=csv_default),
        DeclareLaunchArgument('frame_id', default_value='map'),

        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'magnetic_declination_radians': 0.0,
                'yaw_offset': 0.0, # Novatel in Sim usually has 0 offset
                'publish_filtered_gps': True,
                'broadcast_cartesian_transform': True,
                'wait_for_datum': False,
            }],
            remappings=[
                ('/imu/data', '/a200_0000/sensors/imu_0/data'),
                ('/gps/fix', '/a200_0000/sensors/gps_0/fix'),
                ('/odometry/filtered', '/odometry/global')
            ]
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            output='screen',
            parameters=[ekf_local_params],
            remappings=[('/odometry/filtered', '/odometry/local')]
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            output='screen',
            parameters=[ekf_global_params],
            remappings=[('/odometry/filtered', '/odometry/global')]
        ),

        Node(
            package='waypoint_nav',
            executable='path_publisher',
            name='path_publisher',
            parameters=[{
                'csv_file': LaunchConfiguration('csv_file'),
                'frame_id': LaunchConfiguration('frame_id'),
                'publish_rate': 2.0,
                'use_sim_time': use_sim_time,
            }],
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])