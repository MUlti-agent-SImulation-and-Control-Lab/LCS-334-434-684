"""
Phase 2 combined launch — bypasses Crazyswarm2 entirely.

Stack:
    cflib_bridge_node (cflib direct -> cf2)  publishes /cf_1/odom
    et_tube_mpc_node                         publishes /drone/cmd_accel
    waypoint_publisher_node                  publishes /waypoint
    tube_viz_node                            publishes /tube_viz

cflib_bridge_node owns the CFLib connection, does the takeoff at startup,
and forwards /drone/cmd_accel as full-state setpoints. No Crazyswarm2 in
the loop.

Run after:
  Terminal A: ~/mpc_ws/phase1/launch_sitl.sh        (gz sim + cf2)
Then:
  Terminal B: ros2 launch lcs phase2_bringup.launch.py trajectory:=M

The cflib_bridge_node's __init__ blocks ~5 s while it connects, resets the
estimator, and takes off; once it prints "CFLIB BRIDGE READY" the MPC stack
already has odom flowing and starts driving the drone toward waypoints.

Args:
  drone_name (default: cf_1)
  trajectory (default: M; 'M' or 'S')
  loop (default: false)
  trajectories_path (default: /root/mpc_ws/phase2/trajectories.py)
  start_delay (default: 0.0)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    drone_name = LaunchConfiguration('drone_name')
    trajectory = LaunchConfiguration('trajectory')
    loop_arg = LaunchConfiguration('loop')
    trajectories_path = LaunchConfiguration('trajectories_path')
    start_delay = LaunchConfiguration('start_delay')

    odom_remap = ('/drone/odom', ['/', drone_name, '/odom'])

    declared_args = [
        DeclareLaunchArgument('drone_name', default_value='cf_1'),
        DeclareLaunchArgument('trajectory', default_value='M'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('trajectories_path',
                              default_value='/root/mpc_ws/phase2/trajectories.py'),
        DeclareLaunchArgument('start_delay', default_value='0.0'),
        DeclareLaunchArgument('takeoff_height', default_value='1.0'),
    ]

    cflib_bridge = Node(
        package='lcs',
        executable='cflib_bridge_node',
        name='cflib_bridge_node',
        output='screen',
        parameters=[{
            'uri': 'udp://127.0.0.1:19850',
            'drone_name': drone_name,
            'frame_id': 'world',
            'child_frame_id': 'cf_1',
            'takeoff_height': LaunchConfiguration('takeoff_height'),
            'takeoff_duration': 2.5,
            'estimator_settle': 2.0,
            'odom_period_ms': 20,
            'dt': 0.05,
            'warmup_enabled': False,
        }],
        # cflib_bridge publishes /<drone_name>/odom natively — no remap needed.
    )

    mpc_node = Node(
        package='lcs',
        executable='et_tube_mpc_node',
        name='et_tube_mpc_node',
        output='screen',
        parameters=[{
            'horizon': 20,
            'dt': 0.05,
            'control_rate': 20.0,
            'waypoint_threshold': 0.15,
            'trigger_threshold': 0.15,
            'consecutive_threshold': 3,
            'nominal_lookahead': 10,
            'frame_id': 'world',
        }],
        remappings=[odom_remap],
    )

    waypoint_node = Node(
        package='lcs',
        executable='waypoint_publisher_node',
        name='waypoint_publisher_node',
        output='screen',
        parameters=[{
            'trajectory': trajectory,
            'trajectories_path': trajectories_path,
            'publish_rate': 1.0,
            'waypoint_delay': 2.0,
            'waypoint_threshold': 0.15,
            'loop': loop_arg,
            'start_delay': start_delay,
        }],
        remappings=[odom_remap],
    )

    tube_viz = Node(
        package='lcs',
        executable='tube_viz_node',
        name='tube_viz_node',
        output='screen',
        parameters=[{
            'frame_id': 'world',
            'publish_rate': 10.0,
            'trigger_lifetime_s': 2.0,
            'cylinder_height_clip': 0.5,
        }],
        remappings=[odom_remap],
    )

    return LaunchDescription(declared_args + [cflib_bridge, mpc_node, waypoint_node, tube_viz])
