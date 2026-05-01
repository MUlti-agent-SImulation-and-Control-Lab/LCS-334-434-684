import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    # 1. Get Arguments
    track_name = LaunchConfiguration('track_name').perform(context)
    controller_type = LaunchConfiguration('controller').perform(context).lower().strip()
    force_fallback = LaunchConfiguration('force_fallback').perform(context)
    duration = LaunchConfiguration('duration').perform(context)
    exp_name = LaunchConfiguration('experiment_name').perform(context)
    use_ancillary = LaunchConfiguration('use_ancillary').perform(context).lower().strip() == 'true'
    
    # DEBUG
    print(f"[LAUNCH_DEBUG] controller_type: {controller_type}", flush=True)
    
    pkg_ugv = get_package_share_directory('ugv_uav_control')
    pkg_bot = get_package_share_directory('turtlebot3_gazebo')
    pkg_gz = get_package_share_directory('ros_gz_sim')

     # 2. Dynamic Ground Plane Spawning
    sdf_path = os.path.join(pkg_ugv, 'models', 'path_ground_plane_v2', 'model.sdf')
    with open(sdf_path, 'r') as f:
        sdf_content = f.read()
    
    # Replace default track with user argument
    modified_sdf = sdf_content.replace('track4.png', track_name)
    # WRITE TO TEMP FILE (The Fix)
    import tempfile
    temp_sdf_path = os.path.join(tempfile.gettempdir(), 'temp_ground_plane.sdf')
    with open(temp_sdf_path, 'w') as f:
        f.write(modified_sdf)
    
    spawn_ground = Node(package='ros_gz_sim', executable='create',
        arguments=['-name', 'path_ground_plane', '-file', temp_sdf_path, '-z', '-0.05'],
        output='screen')
    
    # 3. Standard Robot Spawning
    robot_sdf = os.path.join(pkg_ugv, 'models', 'turtlebot3_burger_aruco', 'model.sdf')
    spawn_robot = Node(package='ros_gz_sim', executable='create',
        arguments=['-name', 'turtlebot3_burger', '-file', robot_sdf, '-z', '0.01'],
        output='screen')

    # 4. Spawn Drone at 5m height
    drone_sdf = os.path.join(pkg_ugv, 'models', 'x500_downward_camera', 'model.sdf')
    spawn_drone = Node(package='ros_gz_sim', executable='create',
        arguments=['-name', 'uav', '-file', drone_sdf, '-z', '5.0', '-x', '0.0', '-y', '0.0'],
        output='screen')

    # 5. Bridges
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/drone/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/drone/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/model/uav/pose@geometry_msgs/msg/Pose@gz.msgs.Pose',
            '/model/uav/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
        ],
        remappings=[
            ('/drone/camera/image', '/overhead_camera/image'),
            ('/drone/camera/camera_info', '/overhead_camera/camera_info')
        ],
        output='screen')

    # # 4. Bridges
    # bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
    #     arguments=['/overhead_camera/image@sensor_msgs/msg/Image@gz.msgs.Image', 
    #                '/overhead_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'],
    #     output='screen')
        
    robot_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={os.path.join(pkg_bot, "params", "turtlebot3_burger_bridge.yaml")}'],
        output='screen')

    

    # 5. Smart Controller Selection
    nodes_to_start = [spawn_ground, spawn_robot, bridge, robot_bridge, spawn_drone]
    
    # Perception: Original all-in-one node (ArUco + Path + Debug Window)
    nodes_to_start.append(Node(package='ugv_uav_control', executable='middle_path_follower', output='screen'))
    # Note: middle_path_follower publishes /aruco_pose, /waypoints, /middle_path/debug_image

    nodes_to_start.append(Node(package='ugv_uav_control', executable='ekf_node', output='screen',
                               parameters=[{
                                   'block_aruco': LaunchConfiguration('block_aruco'),
                                   'block_aruco_start': LaunchConfiguration('block_aruco_start'),
                                   'block_aruco_end': LaunchConfiguration('block_aruco_end'),
                               }],
                               remappings=[('/uav_gps_pose', '/model/uav/pose')]))
    nodes_to_start.append(Node(package='ugv_uav_control', executable='uav_mpc_node', output='screen',
                               parameters=[{
                                   'force_fallback': (force_fallback.lower() == 'true'),
                                   'z_opt': LaunchConfiguration('z_opt'),
                                   'z_min': LaunchConfiguration('z_min'),
                                   'z_max': LaunchConfiguration('z_max'),
                                   'vz_max': LaunchConfiguration('vz_max'),
                                   'w_alt': LaunchConfiguration('w_alt'),
                                   'sigma_z': LaunchConfiguration('sigma_z'),
                                   'use_timer': LaunchConfiguration('use_timer')
                               }],
                               remappings=[('/uav_gps_pose', '/model/uav/pose'),
                                           ('/uav_cmd_vel', '/model/uav/cmd_vel')]))
                                           
    nodes_to_start.append(Node(package='ugv_uav_control', executable='experiment_logger', output='screen',
                               parameters=[{'experiment_name': exp_name, 'duration': float(duration)}],
                               remappings=[('/uav_gps_pose', '/model/uav/pose')]))
    
    if controller_type == 'stanley':
        nodes_to_start.append(Node(package='ugv_uav_control', executable='stanley_node', output='screen'))
    elif controller_type == 'pure_pursuit':
        nodes_to_start.append(Node(package='ugv_uav_control', executable='pure_pursuit_node', output='screen'))
    elif controller_type == 'mpc':                                                                         
        nodes_to_start.append(Node(package='ugv_uav_control', executable='mpc_node', output='screen'))
    elif controller_type == 'belief_mpc':
        nodes_to_start.append(Node(package='ugv_uav_control', executable='belief_mpc_node', output='screen',
                                   remappings=[('/uav_gps_pose', '/model/uav/pose')])) 

    # Ancillary Controller (High-frequency correction)
    if use_ancillary:
        nodes_to_start.append(Node(
                package='ugv_uav_control',
                executable='uav_ancillary_node',
                name='uav_ancillary_node',
                parameters=[{
                    'alpha_min': LaunchConfiguration('alpha_min'),
                    'alpha_max': LaunchConfiguration('alpha_max'),
                    'tr_min': LaunchConfiguration('tr_min'),
                    'tr_max': LaunchConfiguration('tr_max'),
                    'tr_spike_threshold': LaunchConfiguration('tr_spike_threshold'),
                    'kappa_thresh': LaunchConfiguration('kappa_thresh'),
                    'drift_threshold': LaunchConfiguration('drift_threshold'),
                    'T_min': LaunchConfiguration('T_min'),
                }],
                remappings=[('/uav_gps_pose', '/model/uav/pose')],
                output='screen'
            ))

    return nodes_to_start

def generate_launch_description():
    pkg_ugv = get_package_share_directory('ugv_uav_control')
    pkg_gz = get_package_share_directory('ros_gz_sim')
    
    # Define Arguments
    arg_track = DeclareLaunchArgument('track_name', default_value='track4.png', description='Texture file to usage')
    arg_ctrl = DeclareLaunchArgument('controller', default_value='belief_mpc', description='Controller node to launch')
    arg_fallback = DeclareLaunchArgument('force_fallback', default_value='false', description='Force UAV fallback policy')
    arg_dur = DeclareLaunchArgument('duration', default_value='60.0', description='Experiment duration in seconds')
    arg_name = DeclareLaunchArgument('experiment_name', default_value='study_run', description='Experiment name')
    arg_use_timer = DeclareLaunchArgument('use_timer', default_value='false', description='Enable 1Hz timer in MPC (legacy)')
    arg_use_ancillary = DeclareLaunchArgument('use_ancillary', default_value='true', description='Launch Ancillary node')

    # New 3D Trajectory Parameters
    arg_z_opt = DeclareLaunchArgument('z_opt', default_value='6.0')
    arg_z_min = DeclareLaunchArgument('z_min', default_value='3.0')
    arg_z_max = DeclareLaunchArgument('z_max', default_value='10.0')
    arg_vz_max = DeclareLaunchArgument('vz_max', default_value='1.0')
    arg_w_alt = DeclareLaunchArgument('w_alt', default_value='50.0')
    arg_sigma_z = DeclareLaunchArgument('sigma_z', default_value='3.0')

    # Phase 2: Tube MPC / Event Trigger Params
    arg_alpha_min = DeclareLaunchArgument('alpha_min', default_value='0.5')
    arg_alpha_max = DeclareLaunchArgument('alpha_max', default_value='2.0')
    arg_tr_min = DeclareLaunchArgument('tr_min', default_value='0.005')
    arg_tr_max = DeclareLaunchArgument('tr_max', default_value='0.15')
    arg_tr_spike = DeclareLaunchArgument('tr_spike_threshold', default_value='0.12')
    arg_kappa = DeclareLaunchArgument('kappa_thresh', default_value='0.4')
    arg_drift = DeclareLaunchArgument('drift_threshold', default_value='0.07')
    arg_tmin = DeclareLaunchArgument('T_min', default_value='3.0')
    
    # Sanity Check Arguments
    arg_block = DeclareLaunchArgument('block_aruco', default_value='false')
    arg_bstart = DeclareLaunchArgument('block_aruco_start', default_value='25.0')
    arg_bend = DeclareLaunchArgument('block_aruco_end', default_value='30.0')

    # Environment
    gz_sim_path = SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', 
        value=[os.path.join(pkg_ugv, 'models'), ':', os.environ.get('GZ_SIM_RESOURCE_PATH', '')])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gz, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {os.path.join(pkg_ugv, "worlds", "path_following.world")}'}.items(),
    )

    return LaunchDescription([
        arg_track,
        arg_ctrl,
        arg_fallback,
        arg_dur,
        arg_name,
        arg_use_timer,
        arg_use_ancillary,
        arg_z_opt,
        arg_z_min,
        arg_z_max,
        arg_vz_max,
        arg_w_alt,
        arg_sigma_z,
        arg_alpha_min,
        arg_alpha_max,
        arg_tr_min,
        arg_tr_max,
        arg_tr_spike,
        arg_kappa,
        arg_drift,
        arg_tmin,
        arg_block,
        arg_bstart,
        arg_bend,
        gz_sim_path,
        gz_sim,
        OpaqueFunction(function=launch_setup)
    ])