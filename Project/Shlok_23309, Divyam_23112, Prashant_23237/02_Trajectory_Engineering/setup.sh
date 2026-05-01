#!/usr/bin/env bash
# Native install for CrazySim ET-Tube MPC on Ubuntu 22.04 + ROS 2 Humble.
#
# Docker is the easier path — see README.md.  This script is for users who
# really want a host-side install (e.g. to develop on the firmware).
#
# Idempotent-ish: rerunning is safe; apt + pip skip already-installed packages.
# Run from the repo root after `git clone --recursive`.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
echo "[setup] repo root: $REPO"

# ----- 1. Apt deps -----
echo "[setup] apt deps"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential cmake git tmux \
    python3 python3-pip python3-dev \
    libcdd-dev libgmp-dev \
    ros-humble-desktop \
    ros-dev-tools

# ros-humble-gazebo bindings (Gazebo Sim Garden 7.x)
sudo apt-get install -y --no-install-recommends \
    ros-humble-ros-gz \
    libgz-sim7-dev libgz-msgs9-dev libgz-transport12-dev || true

# ----- 2. Submodules -----
echo "[setup] submodules"
git -C "$REPO" submodule update --init --recursive

# ----- 3. Python deps -----
echo "[setup] pip deps"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$REPO/requirements.txt"

# CrazySim cflib fork (NOT pip cflib — has the threaded UDP receiver + scan_interface)
echo "[setup] CrazySim cflib fork"
python3 -m pip install --no-deps --upgrade --ignore-installed "$REPO/CrazySim/crazyflie-lib-python"

# transforms3d apt install is broken on numpy 2.x; force pip override
python3 -m pip install --upgrade --ignore-installed transforms3d

# ----- 4. Apply our patched crazyflies.yaml -----
echo "[setup] patching crazyflies.yaml"
TARGET="$REPO/CrazySim/crazyswarm2_ws/src/crazyswarm2/crazyflie/config/crazyflies.yaml"
if [ -f "$TARGET" ]; then
    cp "$REPO/configs/crazyflies.yaml" "$TARGET"
    echo "[setup]   patched: $TARGET"
fi

# ----- 5. Build CrazySim firmware (cf2 SITL binary + Gazebo plugin) -----
echo "[setup] building CrazySim firmware"
mkdir -p "$REPO/CrazySim/crazyflie-firmware/sitl_make/build"
( cd "$REPO/CrazySim/crazyflie-firmware/sitl_make/build" && cmake .. && make -j"$(nproc)" )

# ----- 6. Build crazyswarm2 workspace -----
echo "[setup] building crazyswarm2_ws"
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
( cd "$REPO/CrazySim/crazyswarm2_ws" && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release )

# ----- 7. Build mpc_ws (our package) -----
echo "[setup] building mpc_ws"
# shellcheck disable=SC1091
source "$REPO/CrazySim/crazyswarm2_ws/install/setup.bash"
( cd "$REPO/mpc_ws" && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release )

cat <<MSG

[setup] DONE.

Sanity check (in a fresh shell):
  source /opt/ros/humble/setup.bash
  source $REPO/CrazySim/crazyswarm2_ws/install/setup.bash
  source $REPO/mpc_ws/install/setup.bash
  ros2 pkg list | grep lcs

To run, you'll want the helper commands in the published Docker image
(/usr/local/bin/phase1, phase2, stop_sitl). To replicate them locally,
adapt the scripts under mpc_ws/phase1/ and mpc_ws/phase2/.

Easier path:  use the Docker image — see README.md.
MSG
