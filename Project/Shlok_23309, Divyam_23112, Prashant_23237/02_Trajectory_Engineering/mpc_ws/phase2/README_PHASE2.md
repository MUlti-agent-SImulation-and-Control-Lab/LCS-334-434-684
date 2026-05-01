# Phase 2 — ET-Tube MPC Integration (M / S trajectories)

## Quick start

Inside the container:

```bash
phase2          # M trajectory (default)
phase2 S        # S trajectory
```

This launches a tmux session with 5 auto-starting windows. After Phase 2 is done:

```bash
tmux kill-session -t phase2
```

For Phase 1 alone (hover only, no MPC):

```bash
phase1
```

(Both launchers live at `/usr/local/bin/phase{1,2}`. They use tmux 3.2a.)

### Tmux navigation
- `Ctrl+B 0/1/2/3/4` switch windows
- `Ctrl+B d` detach (session keeps running; `tmux attach -t phase2` to reattach)
- `Ctrl+B :` then `kill-session` for force-stop

## What runs where

```
waypoint_publisher_node ──▶ /waypoint
                                │
                                ▼
                    et_tube_mpc_node ──▶ /drone/cmd_accel
                          ▲   │ │
                          │   │ ├──▶ /drone/nominal_pose ──┐
                          │   │ └──▶ /drone/et_trigger ────┤
                  /cf_1/odom  │                            ▼
                          │   ▼                        tube_viz_node ──▶ /tube_viz
                          │ crazyswarm_bridge_node ──▶ /cf_1/cmd_full_state
                          │                                  │
                  ┌── crazyflie_server.py (Python; backend:=cflib) ────────────────┐
                  │                ▲                                                │
                  │      CFLib over UDP 127.0.0.1:19850                              │
                  │                ▼                                                │
                  └─────── cf2 firmware (sitl_make/build/cf2)  ◀────────────────── ┘
                                ▲
                          internal protocol
                                ▼
                    gz sim Garden + libgz_crazysim_plugin
```

`drone_name` is `cf_1` everywhere (crazyflies.yaml key, launch arg, ROS topic prefix).

## Window timing (phase2 launcher)

| Window | Name | t (s) | Command |
|---|---|---|---|
| 0 | A-sitl | 0 | `~/mpc_ws/phase1/launch_sitl.sh` |
| 1 | C-hover | 15 | `python3 ~/mpc_ws/phase2/hover_hold.py` |
| 2 | B-cswarm | 35 | `ros2 launch crazyflie launch.py backend:=cflib gui:=False teleop:=False mocap:=False` |
| 3 | D-mpc | 55 | `ros2 launch lcs phase2_bringup.launch.py trajectory:=M` |
| 4 | E-rviz | 60 | `ros2 run rviz2 rviz2 -d ~/mpc_ws/phase2/phase2.rviz` |

Sleep durations are tuned for ~94% RTF (X11 software rendering inside Docker). Adjust the sleeps in `/usr/local/bin/phase2` if your sim runs slower/faster.

## Why `gui:=False teleop:=False mocap:=False`

These three Crazyswarm2 nodes are launched by default but break or are unneeded in our SITL setup:

- **gui (gui.py)** crashes on startup because the apt-installed matplotlib was compiled against numpy 1.x and we have numpy 2.2.6 (`AttributeError: _ARRAY_API not found`), and because the apt-pinned nicegui 1.4.2 lacks the `follow_symlink` kwarg the bundled gui.py uses.
- **teleop** wants a joystick; not relevant here.
- **mocap (motion_capture_tracking_node)** is for external mocap; we use simulated odometry from the firmware itself.

Disabling all three is the minimal fix. The actual `crazyflie_server.py` (the only thing we need from the launch) is still launched.

## Manual run sequence (if you want to understand each step)

If you ever need to bypass the tmux launcher and run by hand:

```bash
# Terminal A
~/mpc_ws/phase1/launch_sitl.sh

# Terminal C (after Gazebo GUI is up + the unpause line printed)
python3 ~/mpc_ws/phase2/hover_hold.py
# wait until you see:  "CFLib disconnected - safe for Crazyswarm2"

# Terminal B (only AFTER the disconnect banner)
source /opt/ros/humble/setup.bash
source ~/CrazySim/crazyswarm2_ws/install/setup.bash
ros2 launch crazyflie launch.py backend:=cflib gui:=False teleop:=False mocap:=False

# Terminal D (after Crazyswarm2 logs "Connection to udp://127.0.0.1:19850 Established")
source /opt/ros/humble/setup.bash
source ~/CrazySim/crazyswarm2_ws/install/setup.bash
source ~/mpc_ws/install/setup.bash
ros2 launch lcs phase2_bringup.launch.py trajectory:=M

# RViz (any free terminal)
source /opt/ros/humble/setup.bash
ros2 run rviz2 rviz2 -d ~/mpc_ws/phase2/phase2.rviz
```

## Troubleshooting

### `et_tube_mpc_node` keeps logging "No odometry received yet"

This means `/cf_1/odom` is not being published. Check, in any terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/CrazySim/crazyswarm2_ws/install/setup.bash
ros2 topic list | grep cf_1
ros2 topic hz /cf_1/odom
```

If the topic does not exist:
- The Crazyswarm2 server probably failed to connect to cf2. Look at the B-cswarm window for stack traces. Common causes: cf2 not yet running, hover_hold still holding the socket, or a previous crazyflie_server.py instance lingering.
- Verify `~/CrazySim/crazyswarm2_ws/src/crazyswarm2/crazyflie/config/crazyflies.yaml` has `cf_1: { uri: udp://127.0.0.1:19850 }` and that `cf_sim` type has `firmware_logging.enabled: true` and `default_topics.odom.frequency: 100`.

If the topic exists but is at 0 Hz: the log block was created but cf2 isn't streaming. Restart the SITL window (kill `cf2`, re-run `launch_sitl.sh`). Phase 1 doc Bug #4: cf2 retains state across CFLib disconnects.

### gui.py / matplotlib / nicegui errors in window B

You forgot `gui:=False` in the Crazyswarm2 launch. The phase2 launcher already passes the correct flags; if you ran by hand, add them.

### Drone visibly drifts off the M/S corners

The MPC `waypoint_threshold` is 0.15 m. The waypoint publisher advances at the same threshold but with a 2 s `waypoint_delay`. If you want tighter corners, lower both thresholds in `phase2_bringup.launch.py` parameters and rebuild (`colcon build --symlink-install`).

## Topics quick reference

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/cf_1/odom` | nav_msgs/Odometry | crazyswarm2 → MPC, bridge, viz | only published when `default_topics.odom.frequency: 100` is set |
| `/cf_1/cmd_full_state` | crazyflie_interfaces/FullState | bridge → crazyswarm2 | reactive; rate matches MPC ~20 Hz |
| `/drone/cmd_accel` | geometry_msgs/Accel | MPC → bridge | not remapped; literal |
| `/drone/nominal_pose` | geometry_msgs/PoseStamped | MPC → tube_viz | x_bar[0] each solve |
| `/drone/mpc_status` | std_msgs/String | MPC → debug | per-step solver status |
| `/drone/et_trigger` | std_msgs/String | MPC → tube_viz | event triggers |
| `/waypoint` | geometry_msgs/Point | waypoint pub → MPC, viz | one Point per WP |
| `/tube_viz` | visualization_msgs/MarkerArray | tube_viz → RViz | 10 Hz |

## Build / rebuild

After any source change in `~/mpc_ws/src/lcs_tube/`:

```bash
cd ~/mpc_ws
source /opt/ros/humble/setup.bash
source ~/CrazySim/crazyswarm2_ws/install/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Verify:

```bash
source ~/mpc_ws/install/setup.bash
ros2 pkg list | grep lcs
ros2 pkg executables lcs
# expect: et_tube_mpc_node, crazyswarm_bridge_node, waypoint_publisher_node,
#         tube_viz_node, plus older standard_mpc/tube_mpc/gazebo_bridge.
```

## Carry-over from Phase 1

- `cflib` is the CrazySim fork at `~/CrazySim/crazyflie-lib-python/`, installed via `pip install --no-deps`.
- `setuptools 79.0.1` (downgraded from 82 to satisfy colcon `<80`).
- Firmware must be fully restarted between hover runs.
- One stale apt source list (`gazebo.list`) was disabled to install tmux: `/etc/apt/sources.list.d/gazebo.list.disabled`.
