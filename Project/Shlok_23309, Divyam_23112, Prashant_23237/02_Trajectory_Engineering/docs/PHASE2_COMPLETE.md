# Phase 2 — Complete

ET-Tube MPC drives a Crazyflie through `M` and `S` shaped waypoint trajectories in CrazySim Gazebo SITL with the mRPI tube visualised live in RViz2. Both trajectories complete end-to-end, every waypoint reached, no bound violations.

> **Phase 2 gate:** drone takes off, follows trajectory, MPC publishes triggers / nominal trajectory, tube viz renders, RViz subscribes.

---

## 🏗️ Architecture (final)

Crazyswarm2 is **not** used in Phase 2. After diagnosing that cf2 SITL refuses to actuate motors via Crazyswarm2's `cmd_full_state` path, the design pivoted to a direct CFLib bridge.

```
SITL: gz sim Garden + libgz_crazysim_plugin + cf2 firmware
         ↕ CFLib UDP 127.0.0.1:19850 (single persistent client)
cflib_bridge_node       (lcs.cflib_bridge_node)
  - Owns the CFLib connection for the whole run
  - Kalman reset + takeoff (high_level_commander) on startup
  - Log block at 50 Hz → /cf_1/odom
  - Forwards /drone/cmd_accel + /drone/nominal_state
    → cf.commander.send_full_state_setpoint(pos, vel, acc, ...)
         ↕ /drone/cmd_accel, /drone/nominal_state
et_tube_mpc_node        (lcs.et_tube_mpc_node)
  - Subscribes /cf_1/odom (remapped from /drone/odom), /waypoint
  - Publishes /drone/cmd_accel, /drone/nominal_pose, /drone/nominal_state,
    /drone/et_trigger, /drone/mpc_status
  - Horizon N=20, dt=0.05 s, control rate 20 Hz
  - Trigger threshold 0.15 m, consecutive 3 steps
  - Nominal lookahead: x_traj[10] (500 ms ahead in MPC plan)
         ↕ /waypoint
waypoint_publisher_node (lcs.waypoint_publisher_node)
  - Loads named trajectory (M / S / ...) via importlib
  - Advances when within 0.15 m of current target
         ↓ /tube_viz
tube_viz_node           (lcs.tube_viz_node)
  - mRPI cylinder around nominal pose (LP-projected from Omega_H/Omega_h)
  - Current pose / nominal pose / waypoints / triggers as markers
  - Persistent markers (lifetime=0) except triggers (2 s fade)
```

---

## ▶️ Working Run Sequence

Single command (recommended):

```bash
phase2          # M trajectory (default)
phase2 M
phase2 S
```

This launches a tmux session with three windows:

| Window       | Auto-start at | Process                                                                                          |
| ------------ | ------------- | ------------------------------------------------------------------------------------------------ |
| `0  A-sitl`  | t=0  s        | `~/mpc_ws/phase1/launch_sitl.sh X Y` (X,Y from `trajectory[0]`)                                  |
| `1  B-mpc`   | t=12 s        | `ros2 launch lcs phase2_bringup.launch.py trajectory:=… takeoff_height:=… start_delay:=12.0`     |
| `2  C-rviz`  | t=20 s        | `ros2 run rviz2 rviz2 -d ~/mpc_ws/phase2/phase2.rviz`                                            |

Drone is spawned at the trajectory's first waypoint `xy`. Takeoff height is that waypoint's `z`. RViz comes up while the drone hovers at the start. Trajectory tracking starts at t=24 s (4 s after RViz). Total time from `phase2 X` to first waypoint motion: **~24 s**.

**Stop everything:** `stop_sitl`.

---

## 🔧 What Was Built

| Path | Purpose |
| ---- | ------- |
| `~/mpc_ws/src/lcs_tube/lcs/cflib_bridge_node.py` | **new** — direct CFLib bridge owning the SITL connection, takeoff, odom log block, full-state setpoint forwarding |
| `~/mpc_ws/src/lcs_tube/lcs/et_tube_mpc_node.py` | modified — added `/drone/nominal_state` (Odometry pos+vel), `/drone/nominal_pose` (PoseStamped). Aligned BEST_EFFORT QoS. |
| `~/mpc_ws/src/lcs_tube/lcs/waypoint_publisher_node.py` | modified — accepts `trajectory` + `trajectories_path` params; loads named trajectory via importlib. |
| `~/mpc_ws/src/lcs_tube/lcs/tube_viz_node.py` | **new** — mRPI cylinder via scipy LP projection of Omega; current pose / nominal pose / waypoints / triggers markers. |
| `~/mpc_ws/src/lcs_tube/launch/phase2_bringup.launch.py` | **new** — combined launch (cflib_bridge + MPC + waypoint pub + viz). Args: `drone_name`, `trajectory`, `takeoff_height`, `start_delay`, `loop`, `trajectories_path`. |
| `~/mpc_ws/src/lcs_tube/setup.py` | added `cflib_bridge_node` and `tube_viz_node` entry points; added `phase2_bringup.launch.py` to `data_files`. |
| `~/mpc_ws/src/lcs_tube/simulations/tube_calculation.py` | modified — arena bounds widened to 5×5×3 m; velocity caps to ±2.5 m/s. |
| `~/mpc_ws/src/lcs_tube/simulations/tube_data.npz` | regenerated with new bounds. mRPI radius unchanged at 0.408 m (depends on K and W, not arena). |
| `~/mpc_ws/phase1/launch_sitl.sh` | accepts positional `X Y` for spawn position (default `0 0`). |
| `~/mpc_ws/phase2/trajectories.py` | M / S waypoint definitions in a `TRAJECTORIES` dict. |
| `~/mpc_ws/phase2/phase2.rviz` | RViz config: world frame, MarkerArray on `/tube_viz`, Odometry on `/cf_1/odom`, Pose on `/drone/nominal_pose`. |
| `~/mpc_ws/phase2/hover_hold.py` | optional CFLib warm-up script (legacy; not in normal flow). |
| `~/mpc_ws/phase2/README_PHASE2.md` | Phase 2 user guide. |
| `/usr/local/bin/phase2` | tmux launcher; reads trajectory's first waypoint, sets spawn xy and takeoff height. |
| `/usr/local/bin/stop_sitl` | kills SITL + MPC stack + RViz + tmux sessions. |

---

## 🐞 Bugs Hit and Fixes Applied

### 1. crazyswarm2 `gui.py` crashed on launch

**Cause:** apt `python3-matplotlib` was compiled against numpy 1.x → `AttributeError: _ARRAY_API not found` under our numpy 2.2.6. Plus apt `nicegui 1.4.2` doesn't accept the `follow_symlink` kwarg the bundled `gui.py` passes to `App.add_static_files()`.

**Fix:** disable optional crazyswarm2 nodes — pass `gui:=False teleop:=False mocap:=False` to `ros2 launch crazyflie launch.py`. (Later abandoned crazyswarm2 entirely; see #3.)

---

### 2. crazyswarm2 `crazyflie_server.py` crashed on startup

**Cause:** apt `transforms3d 0.3.1` does `_MAX_FLOAT = np.maximum_sctype(np.float)` at module level. Both `np.maximum_sctype` and `np.float` were removed in numpy 2.0, so the import fails. `tf_transformations` → `transforms3d` chain dies before `crazyflie_server` can connect to cf2. **Symptom:** `/cf_1/odom` topic never appeared and downstream nodes logged `"No odometry received yet"` forever.

**Fix:**

```bash
pip install --upgrade --ignore-installed transforms3d
```

The `--ignore-installed` is required because pip can't uninstall the apt distutils-installed 0.3.1; the new 0.4.x lands in `/usr/local/lib/python3.10/dist-packages/` and shadows it.

---

### 3. cf2 SITL refuses to actuate motors via Crazyswarm2's path

**Symptom:** Even after #1 and #2 were fixed and `crazyswarm2_server` was streaming `/cf_1/odom` correctly, the `/cf_1/takeoff` service call returned OK but the drone stayed on the ground at z=0.015 m. The bridge published `/cf_1/cmd_full_state` at 20 Hz, crazyswarm2 forwarded to cf2 via CFLib, cf2 didn't move.

**Diagnosis:** CrazySim's cf2 SITL plugin gates motor activation on continuous CFLib client activity. The Crazyswarm2 `cflib backend` connects briefly to issue commands then idles — the plugin treats it as "no active client" and suppresses motor output.

**Fix:** Bypassed Crazyswarm2 entirely. New `cflib_bridge_node.py` opens a single CFLib connection, takes off via `high_level_commander.takeoff()`, and stays connected for the entire run, forwarding MPC commands as `cf.commander.send_full_state_setpoint(...)`.

---

### 4. Bridge prediction caused runaway climb (z reached 8.65 m)

**Cause:** Original `crazyswarm_bridge_node` and the first version of `cflib_bridge_node` predicted the position setpoint forward from current odom: `p_pred = p + v·dt + 0.5·a·dt²`. With MPC commanding `az = +4 m/s²` to climb, the predicted z grows positively each cycle → setpoint keeps rising → drone keeps climbing → `vz` grows → next predicted z is even higher. **Positive feedback runaway.**

**Fix:** Stopped predicting. The bridge now uses the MPC's nominal trajectory directly: position+velocity setpoint comes from `/drone/nominal_state` (which is the MPC's `x_traj[lookahead]`). The MPC internally generates a stable nominal trajectory toward the waypoint; sending it directly tracks the trajectory without positive feedback.

---

### 5. Drone stuck at midpoint of waypoint segments

**Cause:** First fix for #4 sent `vel = 0` (firmware tries to stop at the position setpoint each cycle). With horizon=20 and a 2 m waypoint, the MPC's `x_traj[1]` is only 50 ms ahead — barely above current state. Drone's tracking effort was overwhelmed by the "decelerate to stop" implied by zero velocity. Drone moved 0.02 m/s and stalled before reaching waypoints.

**Fix:** Use the MPC's nominal velocity from `x_traj[1]` (which is non-zero along the trajectory). Added `/drone/nominal_state` (`Odometry` with both pose and twist) so bridge gets velocity feedforward too. Also bumped `nominal_lookahead` parameter to 10 (500 ms ahead) so the position setpoint is far enough forward to drive meaningful motion.

---

### 6. Drone oscillated near arena boundary, never reached waypoints

**Cause:** Original arena bounds (`px ∈ [-1, 1]`, `py ∈ [-1.5, 1.5]`, vel cap ±1.5 m/s) excluded the M and S waypoints (M peaks at `(2, 2)`, S peaks at `(-1, 1.5)` and `(0, 2)`). MPC's QP couldn't find a feasible solution that satisfied state constraints AND reached the waypoint. Solver returned `"Solution may be inaccurate"` warnings (visible in logs); drone oscillated near the boundary. Velocity cap of 1.5 m/s also meant horizon=1s couldn't span 2-meter waypoint distances geometrically.

**Fix:** Widened state constraints in `tube_calculation.py` to `px,py ∈ ±2.5 m`, `pz ∈ [0, 3]`, vel cap `±2.5 m/s`. Re-ran `tube_calculation.py` to regenerate `tube_data.npz`. Tightened arena (after subtracting tube): `px,py ∈ ±2.09 m`. M and S waypoints now fit. Tube radius unchanged at 0.408 m (depends on K and disturbance set, not arena).

---

### 7. `cflib_bridge_node` first `param.set_value` timed out under `ros2 launch`

**Symptom:** Bridge logged `"CFLib link is open"`, then crashed with `Exception: Connection timed out` from `cflib.crazyflie.param.set_value`. Same code path worked when run via `ros2 run` directly or via standalone Python. Only failed under `ros2 launch`.

**Cause:** cf2's parameter TOC isn't ready immediately after `SyncCrazyflie.open_link()` returns under `ros2 launch`'s threading model (subtle interaction between rclpy executor and cflib internal threads). Setting a param within ~5 ms of `open_link` returning hits the cflib param subsystem before its TOC is populated.

**Fix:** Inserted `time.sleep(1.0)` between `open_link()` and the first `param.set_value()`. Empirically reliable. Documented at the call site in `cflib_bridge_node.py`.

---

### 8. QoS mismatch warning on `/drone/et_trigger`

**Cause:** Default Python rclpy QoS is `RELIABLE, VOLATILE, KEEP_LAST(10)`. Default `topic hz` subscriber uses `RELIABLE` too, but our high-rate publishers (`cmd_accel`, `odom`) used `BEST_EFFORT` through Crazyswarm2's pose log. Cross-publisher mismatch.

**Fix:** Aligned all per-step topics to explicit `BEST_EFFORT, VOLATILE, depth=10` QoS via a helper `_qos_be()` factory in `et_tube_mpc_node.py`, `cflib_bridge_node.py`, `tube_viz_node.py`. `/drone/nominal_pose` and `/tube_viz` (event-driven, not high-rate) kept as `RELIABLE` — RViz subscribes to them as `RELIABLE` so it works.

---

### 9. RViz `phase2.rviz` had a broken `RobotModel` display

**Cause:** The Description Topic was set to `/drone/et_trigger` (a `std_msgs/String`) — wrong type for `RobotModel` which expects `std_msgs/String robot_description` URDF text. RViz spammed warnings.

**Fix:** Removed the `RobotModel` display from `phase2.rviz`. Added a `Pose` display for `/drone/nominal_pose` (yellow axes). Final config: Grid + TubeMarkers + CFOdom + NominalPose, fixed frame `world`, orbit camera focused on `(0.5, 1.0, 1.0)`.

---

### 10. Marker lifetime made tube + waypoints flicker

**Cause:** Default `Duration` (sec=0, nanosec=0) is interpreted by RViz as "never expire" only when explicitly set. Without setting, RViz treated stale markers as having expired between MarkerArray re-publishes at 10 Hz.

**Fix:** Persistent markers (cylinder, current pose dot, nominal pose dot, waypoints) explicitly set `m.lifetime = Duration(sec=0, nanosec=0)`. Trigger spheres keep `lifetime = 2.0 s` so they fade out.

---

### 11. Drone wasn't airborne when MPC started (separate from #3)

After abandoning Crazyswarm2 (fix #3), the cflib_bridge needed to handle takeoff itself. Initial design just forwarded `cmd_full_state` from a ground state — bridge sent setpoints near current pose, motors didn't spin enough to lift.

**Fix:** Bridge calls `cf.high_level_commander.takeoff(takeoff_height, takeoff_duration)` in `__init__`, waits `takeoff_duration + 0.5` s, then sets `_is_armed=True`. Only after takeoff does the bridge start forwarding `cmd_accel` as `send_full_state_setpoint`. The `_is_armed` flag is checked in `_cmd_accel_cb` so accel commands during the takeoff phase are dropped.

---

### 12. Trajectory start vertex hardcoded to (0,0,1)

**Cause:** Drone always spawned at `(0,0)` in Gazebo and took off to z=1.0 regardless of which trajectory was running. For trajectories where the first waypoint isn't `(0,0,1)`, the drone needs to traverse to the start before "starting".

**Fix:** Plumbed trajectory start vertex through three layers:

- `/usr/local/bin/phase2` parses the trajectory file, extracts `TRAJECTORIES[NAME][0]` as `(SX, SY, SZ)`.
- `~/mpc_ws/phase1/launch_sitl.sh` accepts positional `X Y` args, forwards to `sitl_singleagent.sh -x X -y Y`.
- `phase2_bringup.launch.py` accepts `takeoff_height` launch arg, routes to cflib_bridge.

So `phase2 M` spawns at `M[0]`'s xy and takes off to `M[0].z`. Adding a new trajectory just means adding to the `TRAJECTORIES` dict — no other changes needed.

---

### 13. RViz opens after trajectory starts, user misses takeoff

**Cause:** Original launcher had RViz at t=35 s but waypoint publisher had `start_delay=0` so trajectory started immediately after takeoff (around t=20 s). User would miss the takeoff in RViz.

**Fix:** Tightened launcher: RViz at t=20 s (right after takeoff), `start_delay=12 s` so first waypoint fires at t=24 s — 4 s after RViz appears. User sees drone hovering at start vertex in RViz, then watches it follow the trajectory.

---

### 14. `mpl_toolkits` namespace conflict (apt vs pip)

**Symptom:** `sim_stress_tests.py` failed with `ImportError: cannot import name 'docstring' from 'matplotlib'` when importing `Axes3D`.

**Cause:** apt `python3-matplotlib` ships `/usr/lib/python3/dist-packages/mpl_toolkits/__init__.py`, making it a regular package. pip matplotlib 3.10 ships `/usr/local/lib/python3.10/dist-packages/mpl_toolkits/` as a namespace contribution (no `__init__.py`). Python's import system prefers the regular package, so apt's stale `mpl_toolkits.mplot3d` (expecting `matplotlib.docstring`, removed in 3.10) shadowed pip's version.

**Fix:** Sidelined apt's whole tree:

```bash
mv /usr/lib/python3/dist-packages/mpl_toolkits /usr/lib/python3/dist-packages/mpl_toolkits.apt-disabled
rm -f /usr/lib/python3/dist-packages/matplotlib-3.5.1-nspkg.pth
```

Now pip's `mpl_toolkits` namespace package wins. `from mpl_toolkits.mplot3d import Axes3D` works.

---

## ✅ Verification

End-to-end pass on both trajectories:

```
phase2 M    →  drone visits all 5 waypoints in ~26 s
              [0,0,1] → [0,2,1] → [1,1,1] → [2,2,1] → [2,0,1]

phase2 S    →  drone visits all 5 waypoints in ~21 s
              [0,0,1] → [1,0.5,1] → [0,1,1] → [-1,1.5,1] → [0,2,1]
```

### Captured screenshots

`phase2 M` mid-run — drone visible in Gazebo (left), RViz (right) shows the full M traced as a cyan trail with the green waypoint markers, the translucent mRPI tube around the nominal pose, and red trigger sphere events:

![phase2 M run](screenshots/m_trajectory_simulation.png)

`phase2 S` mid-run — same setup, S-shape trajectory:

![phase2 S run](screenshots/s_trajectory_simulation.png)

Per-topic rates while running:

```
ros2 topic hz /cf_1/odom           →  ~45 Hz   (target 50, RTF ~92%)
ros2 topic hz /drone/cmd_accel     →  20 Hz
ros2 topic hz /drone/nominal_state →  20 Hz
ros2 topic hz /tube_viz            →  10 Hz
```

ET trigger rate during steady tracking: **~6–8%** of MPC steps. Solve count drops from "every step" (200/200) under standard MPC to **~15–32** over the same horizon under ET-Tube MPC (per `sim_stress_tests.py` output).

mRPI tube cylinder rendered in RViz: radius 0.408 m, height 0.5 m (z-clipped for visibility from raw 1.278 m). LP-projected from the 72-facet mRPI polytope in `tube_data.npz` at `tube_viz_node` startup.

Stress test (`sim_stress_tests.py`) also verified:

| Scenario                              | Bound violations |
| ------------------------------------- | ---------------- |
| A: persistent crosswind               | None ✓           |
| B: 100% payload drop                  | None ✓           |
| C: high-speed catch + gust            | None ✓           |
| D: UWB sensor glitch                  | None ✓           |

---

## 📦 Environment & Dependencies (Phase 2)

### Carry-over from Phase 1
*(already documented in `PHASE1_COMPLETE.md`)*

- Container `crazysim`, Ubuntu 22.04, ROS 2 Humble, Gazebo Sim Garden 7.9.0, Python 3.10.
- cflib (CrazySim fork) at `/usr/local/lib/python3.10/dist-packages/cflib` (`pip install --no-deps ~/CrazySim/crazyflie-lib-python`).
- setuptools 79.0.1, cvxpy 1.7.x, osqp 1.1.x, clarabel 0.11.x.
- cflib URI: `udp://127.0.0.1:19850` (loopback only).

### Added during Phase 2

- `transforms3d 0.4.x` (pip override of apt 0.3.1).
- `matplotlib 3.10.x` (pip override of apt 3.5.1; required for pypoman + numpy 2.x).
- `pypoman 1.2.x` + `pycddlib` + `cvxopt` (for `tube_calculation.py`).
- apt `libcdd-dev libgmp-dev` (build deps for pycddlib).
- apt `tmux 3.2a` (for phase1/phase2 launchers).
- apt `mpl_toolkits` sidelined → `mpl_toolkits.apt-disabled` (let pip win).
- One stale apt source `gazebo.list.disabled` (duplicate of `gazebo-stable.list`; was blocking `apt update`).

---

## ⚠️ Known Limitations / Open Items

- **cf2 SITL warmup:** occasionally cf2's first CFLib client connect doesn't kick its log block streaming into life, and odom stays at 0 Hz. The 1 s wait in #7 covers most cases. The bridge has a `warmup_enabled` parameter (default off in launch, on in source) for an explicit connect-disconnect-reconnect priming cycle if needed.
- **Tracking speed:** drone moves at ~0.6 m/s steady-state on the M/S trajectories. To go faster, raise `nominal_lookahead` (currently 10) or the velocity-tracking weight in `Q`. Untuned but stable.
- **Real-time factor:** Gazebo runs at ~92% RTF under software rendering (`LIBGL_ALWAYS_SOFTWARE=1`). All times in this doc are wall-clock; sim time runs proportionally slower.
- **Crazyswarm2 not in the loop:** We don't use `/cf_1/cmd_full_state`. If a future feature needs Crazyswarm2's services (e.g., motion capture integration), the bypass would have to be revisited.

---

## 🔄 Carry-over for Any Future Phases

- **mRPI tube** radius 0.408 m, height 0.5 m. mRPI computation in `tube_calculation.py` uses 100-step truncation of `Σ A_k^i W` with explicit truncation compensation (~2.6% scaling factor to ensure outer approximation).
- **Disturbance set W** combines acceleration disturbance (`B_d · d`, `d ∈ [±1, ±1, ±1.5]`) with measurement-noise feedback (`-B_d K v_k`, `v ∈ [±0.05, ±0.05, ±0.1]`). Phase 2 simulation must apply `v_k` through the controller, not as additive state noise.
- **Cost matrices:** `Q = diag(10, 10, 10, 1, 1, 1)` (heavy position penalty), `R = diag(1, 1, 1)` for tube calc; `R = diag(0.1, 0.1, 0.1)` for online MPC.
- **Dynamics:** discrete-time double-integrator with `dt=0.05 s`.
- **ET trigger:** position error norm > 0.15 m for 3 consecutive steps → re-solve QP. Otherwise nominal propagation.