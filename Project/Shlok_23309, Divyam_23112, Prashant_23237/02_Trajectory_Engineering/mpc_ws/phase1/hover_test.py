#!/usr/bin/env python3
"""
Phase 1 hover test — robust to Ctrl+C, DEBUG logging.

Connects to cf2 over UDP, resets Kalman, takes off to 1.0 m, hovers 10 s,
lands. If interrupted, sends an emergency land before exit.

Pre-req: launch_sitl.sh running in another terminal (gz sim + cf2 up).
"""
import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = "udp://127.0.0.1:19850"
TAKEOFF_HEIGHT = 1.0
TAKEOFF_TIME = 2.5
HOVER_TIME = 10.0
LAND_TIME = 2.5
EMERGENCY_LAND_TIME = 1.5
ESTIMATOR_SETTLE = 2.0


def safe_land(cf, log, duration: float) -> None:
    try:
        cf.high_level_commander.land(0.0, duration)
        time.sleep(duration + 0.5)
    except Exception as e:
        log.error("land call failed: %s", e)


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("hover_test")

    cflib.crtp.init_drivers()
    log.info("connecting to %s", URI)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="/tmp/cf_cache")) as scf:
        cf = scf.cf
        log.info("connected; TOC + handshake completed")

        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0")
        log.info("kalman reset; settling %.1fs", ESTIMATOR_SETTLE)
        time.sleep(ESTIMATOR_SETTLE)

        cf.param.set_value("commander.enHighLevel", "1")
        time.sleep(0.1)

        airborne = False
        try:
            log.info("takeoff to %.2fm over %.2fs", TAKEOFF_HEIGHT, TAKEOFF_TIME)
            cf.high_level_commander.takeoff(TAKEOFF_HEIGHT, TAKEOFF_TIME)
            airborne = True
            time.sleep(TAKEOFF_TIME + 0.5)

            log.info("hover for %.1fs", HOVER_TIME)
            time.sleep(HOVER_TIME)

            log.info("land over %.2fs", LAND_TIME)
            safe_land(cf, log, LAND_TIME)
            airborne = False
        except KeyboardInterrupt:
            log.warning("INTERRUPTED — attempting emergency land")
            if airborne:
                safe_land(cf, log, EMERGENCY_LAND_TIME)
                airborne = False
        finally:
            if airborne:
                log.warning("exiting while still airborne — emergency land")
                safe_land(cf, log, EMERGENCY_LAND_TIME)
            try:
                cf.high_level_commander.stop()
            except Exception:
                pass

        log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
