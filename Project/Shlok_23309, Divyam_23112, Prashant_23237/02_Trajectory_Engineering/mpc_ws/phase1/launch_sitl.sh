#!/usr/bin/env bash
# Phase 1 launcher: gz sim + cf2 firmware SITL for a single Crazyflie.
# Stack: gz sim Garden -> libgz_crazysim_plugin -> cf2 (UDP 19950) <- CFLib (UDP 19850)
#
# Usage:
#   launch_sitl.sh                  # spawn at (0, 0)
#   launch_sitl.sh X Y              # spawn at (X, Y)
#
# After Gazebo+cf2 are up, we send a defensive WorldControl/pause:false so the
# sim is guaranteed running regardless of GUI defaults. Blocks on the GUI;
# Ctrl-C runs the launcher's cleanup trap.

set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

X="${1:-0}"
Y="${2:-0}"

CRAZYSIM_DIR="${HOME}/CrazySim/crazyflie-firmware"
LAUNCHER="${CRAZYSIM_DIR}/tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_singleagent.sh"
WORLD="crazysim_default"

if [[ ! -f "$LAUNCHER" ]]; then
    echo "ERROR: launcher not found: $LAUNCHER" >&2
    exit 1
fi

(
    for _ in $(seq 1 30); do
        if gz service -l 2>/dev/null | grep -q "/world/${WORLD}/control"; then
            gz service -s "/world/${WORLD}/control" \
                --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
                --timeout 2000 --req "pause: false" >/dev/null 2>&1 || true
            echo "[launch_sitl] sim unpaused (defensive WorldControl call)"
            exit 0
        fi
        sleep 1
    done
    echo "[launch_sitl] WARN: world control service not seen within 30s" >&2
) &

cd "$CRAZYSIM_DIR"
echo "[launch_sitl] spawning Crazyflie at x=$X y=$Y"
exec bash "$LAUNCHER" -m crazyflie -x "$X" -y "$Y"
