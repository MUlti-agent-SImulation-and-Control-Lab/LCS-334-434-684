import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_lcs = get_package_share_directory('lcs')
    urdf_file = os.path.join(pkg_lcs, 'urdf', 'simple_drone.urdf')
    
    # Launch Gazebo server and client
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_pkg, 'launch', 'gzserver.launch.py'))
    )
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_pkg, 'launch', 'gzclient.launch.py'))
    )

    # Spawn the drone
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simple_drone',
            '-file', urdf_file,
            '-x', '0', '-y', '0', '-z', '0.1'
        ],
        output='screen'
    )

    # Bridge node to convert cmd_accel to cmd_force
    bridge_node = Node(
        package='lcs',
        executable='gazebo_bridge_node',
        name='gazebo_bridge_node',
        output='screen'
    )

    # Read URDF to publish robot_state_publisher (optional, but good for RViz)
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    return LaunchDescription([
        gazebo_server,
        gazebo_client,
        robot_state_publisher_node,
        spawn_entity,
        bridge_node
    ])
