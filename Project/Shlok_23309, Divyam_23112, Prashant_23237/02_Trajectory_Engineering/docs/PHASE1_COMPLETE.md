# Phase 1 — Complete

Single Crazyflie takes off, hovers at 1.0 m for 10 s, lands cleanly.
Stack validated: **Gazebo Sim → libgz_crazysim_plugin → cf2 firmware → CFLib (UDP).**

> **No ROS layer in this gate.** Crazyswarm2 server is **not** part of Phase 1.

---

## 🖥️ Environment

| Item | Value |
| ---- | ----- |
| Docker container | `crazysim` (image `crazyy`) |
| OS | Ubuntu 22.04 (jammy) |
| ROS 2 | Humble (`/opt/ros/humble`) — sourced for Phase 2, **not used** in Phase 1 |
| Gazebo | Gazebo Sim Garden 7.9.0 (`/usr/bin/gz`) |
| Python | 3.10 |
| `DISPLAY` | `:0` (X11 socket bind-mounted at `/tmp/.X11-unix`) |
| `LIBGL_ALWAYS_SOFTWARE` | `1` (Mesa software rasterizer; no GPU passthrough) |
| Bind mount | `/media/omen/linux2/crazysim/mpc_code` (host) → `/root/mpc_ws` (container) |

---

## 🔧 What Was Built and Where

| Artifact                    | Location                                                              | Built by                                                                                                             |
| --------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `cf2` firmware SITL binary  | `~/CrazySim/crazyflie-firmware/sitl_make/build/cf2`                   | Pre-built (mtime 2026-02-05)                                                                                         |
| `libgz_crazysim_plugin.so`  | `~/CrazySim/crazyflie-firmware/sitl_make/build/build_crazysim_gz/`    | Pre-built                                                                                                            |
| `crazyswarm2_ws/install/`   | `~/CrazySim/crazyswarm2_ws/install/`                                  | `colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release` (5 packages, 59.6 s, exit 0)                |
| `cflib` (CrazySim fork)     | `/usr/local/lib/python3.10/dist-packages/cflib`                       | `pip install --no-deps ~/CrazySim/crazyflie-lib-python`                                                              |

---

## ▶️ Working Run Sequence

### Terminal A — SITL Bringup (Gazebo + cf2)

```bash
docker exec -it crazysim bash
~/mpc_ws/phase1/launch_sitl.sh
```

`launch_sitl.sh` is a thin wrapper around CrazySim's canonical `sitl_singleagent.sh -m crazyflie -x 0 -y 0`. It:

1. Exports `DISPLAY=:0` and `LIBGL_ALWAYS_SOFTWARE=1` (idempotent).
2. Forks a 30-second watcher that polls `gz service -l` for `/world/crazysim_default/control` and, once it appears, sends one `WorldControl{pause: false}` to guarantee the sim is unpaused.
3. `exec`s `sitl_singleagent.sh -m crazyflie -x 0 -y 0`.

**Expected log signposts (in order):**

- `killing running crazyflie firmware instances`
- `Spawning crazyflie_0 at 0 0`
- `starting instance 0 in /root/CrazySim/crazyflie-firmware/sitl_make/build/0`
- `[launch_sitl] sim unpaused (defensive WorldControl call)` — confirms unpause
- Gazebo GUI window opens with the Crazyflie on the ground plane

---

### Terminal B — CFLib Hover Test

> Run **only after** the Gazebo GUI is up and the unpause line has printed.

```bash
docker exec -it crazysim bash
python3 ~/mpc_ws/phase1/hover_test.py
```

Total runtime ~18 s: connect → 2 s estimator settle → 2.5 s takeoff → 10 s hover at 1.0 m → 2.5 s land. DEBUG logging is enabled; expect to see the full TOC fetch and `connected; TOC + handshake completed` before the takeoff log line.

---

### Shutdown

In Terminal A: close the Gazebo window or press `Ctrl-C`. The trap in `sitl_singleagent.sh` runs `pkill -x cf2` and `pkill -9 ruby` to clean up.

---

## 🐞 Bugs Hit and Fixes Applied

### 1. Gazebo apparently paused (false alarm; defensive fix added anyway)

**Symptom:** The Gazebo GUI showed real-time factor of 91.87% and what appeared to be an active pause indicator. Drone did not move.

**Diagnosis:** cf2 `out.log` showed continuous `ESTKALMAN: WARNING: Kalman prediction rate off (94)` — the firmware was alive and ticking at ~94 Hz against an expected 100 Hz. That mirrors the ~94% RTF, which is the sim running slightly under real-time (X11 software rasterizer + ode physics tax), **not** paused. The "pause icon" in the GUI was the toggle-to-pause button shown when the sim is running.

**Fix in `launch_sitl.sh`:** added a defensive

```bash
gz service -s /world/crazysim_default/control \
    --reqtype gz.msgs.WorldControl --req "pause: false"
```

call once the world control service is reachable. Idempotent; harmless if the sim is already running. Prints `[launch_sitl] sim unpaused (defensive WorldControl call)` on success so state is visible.

The upstream `gz sim -s -r ...` flag (run on start) was already present in `sitl_singleagent.sh`. The `-r` is **not** added to `gz sim -g`; instead we send the WorldControl message at runtime.

---

### 2. Wrong CFLib URI: `0.0.0.0` → `127.0.0.1`

**Symptom:** `hover_test.py` connected, logged `Request _request_protocol_version()`, then hung indefinitely. No takeoff.

**Cause:** Used `udp://0.0.0.0:19850` (typical "any-interface" bind target). The cf2 firmware binds its UDP listener to **loopback only**, so a client connecting to `0.0.0.0` does not reach it.

**Fix:** Change every CFLib URI in scripts and configs to `udp://127.0.0.1:19850`. Single line in `hover_test.py`.

This applies to Phase 2 too: `crazyflies.yaml` URIs must be `udp://127.0.0.1:1985N`, not `udp://0.0.0.0:1985N` (the CrazySim README example uses `0.0.0.0` and is incorrect).

---

### 3. Wrong cflib package: stock pip → CrazySim fork

**Symptom:** Even with the corrected URI, `_request_protocol_version()` still hung. The connect call entered `socket.recvfrom(1024)` and blocked forever.

**Cause:** `pip install cflib` had pulled the upstream Bitcraze cflib (0.1.31). CrazySim ships a fork at `~/CrazySim/crazyflie-lib-python/` with a rewritten UDP driver: threaded receiver, `_BASE_PORT = 19850`, `_NR_OF_PORTS_TO_SCAN = 10`, and `scan_interface` for SITL discovery on ports 19850–19859. The upstream driver opens the socket, sends `\xFF\x01\x01\x01`, then does a blocking single-thread `recvfrom` with no timeout — incompatible with the CrazySim handshake.

**Fix:**

1. `pip uninstall -y cflib`
2. `pip install --upgrade "setuptools>=61,<80" wheel pip` (need ≥61 for the bundled `pyproject.toml` PEP 621 metadata; <80 keeps colcon happy)
3. `pip install --no-deps ~/CrazySim/crazyflie-lib-python` (`--no-deps` avoids pulling the bundle's `packaging~=25.0` and downgrading our existing `packaging==26.0` used by ROS)
4. `pip install -e .` was attempted first but failed: the bundled `pyproject.toml` uses `setuptools.build_meta` without the PEP 660 `build_editable` hook. Non-editable install works.

**Verify the right cflib is loaded** — must run from a directory that does **not** itself contain a `cflib/` subdir, otherwise `sys.path[0]` of `python3 -c` masks a bad install:

```bash
cd / && python3 -c "
import cflib, inspect
from cflib.crtp import udpdriver
print('cflib __file__:', cflib.__file__)
src = inspect.getsource(udpdriver)
assert '_BASE_PORT = 19850' in src, 'WRONG cflib — upstream driver, no scan_interface'
assert 'threading' in src, 'WRONG cflib — no threaded receiver'
print('OK: CrazySim fork active')
"
```

Expected `cflib.__file__`:
`/usr/local/lib/python3.10/dist-packages/cflib/__init__.py`.

---

### 4. Firmware must be fully restarted between runs

**Observed:** A second back-to-back invocation of `hover_test.py` without restarting Terminal A would not produce a working second flight. Killing cf2 and re-running the launcher (which respawns it from scratch) is required.

**Likely cause:** cf2 keeps state across CFLib disconnect/reconnect — including high-level commander state, parameter values, and possibly estimator state. CrazySim's UDP driver is single-client and reuses the socket binding; a clean re-arm needs a fresh process.

**Operational rule:** between hover tests, `Ctrl-C` Terminal A and re-run `~/mpc_ws/phase1/launch_sitl.sh` from scratch. Do not try to re-run `hover_test.py` against a stale cf2.

---

## 📁 Files Written for Phase 1

| Path (container)                | Path (host bind mount)                                              | Purpose                                                          |
| ------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `~/mpc_ws/phase1/launch_sitl.sh` | `/media/omen/linux2/crazysim/mpc_code/phase1/launch_sitl.sh`        | SITL bringup wrapper                                             |
| `~/mpc_ws/phase1/hover_test.py`  | `/media/omen/linux2/crazysim/mpc_code/phase1/hover_test.py`         | CFLib hover test (DEBUG logging, KeyboardInterrupt-safe land)    |

---

## 📝 Open Environment Notes (carry-over to Phase 2)

- **Setuptools** is now **79.0.1** (downgraded from 82.0.1 to satisfy `colcon-core requires setuptools<80`).
- `pip check` still warns:
  ```
  cfclient 2025.12.1 has requirement cflib~=0.1.31,
  but you have cflib 0.0.post1.dev1366+g84f9d5204
  ```
  Cosmetic; `cfclient` is not part of the SITL flow.
- `crazyflies.yaml` still has the upstream radio URI (`radio://0/80/2M/E7E7E7E7E7`) for `cf231`. Phase 2 will replace it with `udp://127.0.0.1:19850` and a drone name aligned with the Crazyswarm2 ROS topic conventions.