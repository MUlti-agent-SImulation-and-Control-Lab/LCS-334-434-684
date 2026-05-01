#!/usr/bin/env python3
"""
Phase 2 CrazySim warm-up.

Connects briefly to cf2 over UDP, resets the Kalman estimator, then disconnects.
Does NOT take off. Reasons:

  1. CrazySim cf2 firmware seems to need an initial CFLib connect-disconnect
     cycle before its log block streaming becomes reliable for subsequent
     clients (notably Crazyswarm2). Without this warm-up, /cf_1/odom comes
     up but stays at 0 Hz.
  2. After CFLib disconnect, cf2 in SITL does NOT keep the drone hovering on
     the high-level commander setpoint — the drone descends. So a takeoff
     here would be undone before Crazyswarm2 takes over anyway.

The actual takeoff is done by the phase2 launcher after Crazyswarm2 is up,
via:  ros2 service call /cf_1/takeoff crazyflie_interfaces/srv/Takeoff ...

Pre-req: launch_sitl.sh running (gz sim + cf2 up).
"""
import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = "udp://127.0.0.1:19850"
ESTIMATOR_SETTLE = 2.0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("hover_hold")

    cflib.crtp.init_drivers()
    log.info("CrazySim warm-up: connecting to %s", URI)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="/tmp/cf_cache")) as scf:
        cf = scf.cf
        log.info("connected; resetting Kalman estimator")
        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(ESTIMATOR_SETTLE)
        log.info("warm-up complete")

    print("=" * 60, flush=True)
    print("CFLib disconnected - safe for Crazyswarm2", flush=True)
    print("=" * 60, flush=True)
    log.info("Drone is on the ground. The phase2 launcher will issue the takeoff via Crazyswarm2 service.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
