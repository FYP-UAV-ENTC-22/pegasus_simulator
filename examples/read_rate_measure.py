#!/usr/bin/env python3
"""
Read roll/pitch + local NED from the flight controller and MEASURE the true rate.

At high requested rates ArduPilot pads the stream with DUPLICATE messages (same
time_boot_ms) to hit the requested Hz, so raw packet rate overstates how much real
information you get. This script reports both, side by side:

  raw   = packets/s actually received
  fresh = packets/s whose time_boot_ms advanced (genuinely new samples)
  dup   = packets/s that repeated the previous time_boot_ms

It also reports the inter-arrival timing of the FRESH samples (mean dt, min/max,
jitter) so you can see how uniform the real 100 Hz is.

Read-only: it requests streams and listens, it never arms or commands anything.

Usage (connect DIRECTLY to SITL, not via MAVProxy, or the rate is throttled):
    python3 read_rate_measure.py --connect tcp:127.0.0.1:5762 --freq 100
"""

import argparse
import math
import time
from collections import defaultdict

from pymavlink import mavutil


class StreamStat:
    """Per-message counters + fresh inter-arrival timing over a 1 s window."""
    def __init__(self):
        self.raw = 0
        self.fresh = 0
        self.dup = 0
        self.last_boot = None
        self.last_fresh_wall = None
        self.dts = []          # wall-clock gaps between fresh samples [s]

    def update(self, boot_ms, now):
        self.raw += 1
        if self.last_boot is None or boot_ms > self.last_boot:
            self.fresh += 1
            if self.last_fresh_wall is not None:
                self.dts.append(now - self.last_fresh_wall)
            self.last_fresh_wall = now
            self.last_boot = boot_ms
        else:
            self.dup += 1               # same (or older) time_boot_ms -> duplicate

    def summary(self, window_s):
        raw_hz = self.raw / window_s
        fresh_hz = self.fresh / window_s
        dup_hz = self.dup / window_s
        dup_pct = (100.0 * self.dup / self.raw) if self.raw else 0.0
        if self.dts:
            mean = sum(self.dts) / len(self.dts)
            lo, hi = min(self.dts), max(self.dts)
            jit = hi - lo
            timing = (f"dt mean={mean*1e3:5.1f}ms min={lo*1e3:5.1f} "
                      f"max={hi*1e3:5.1f} jit={jit*1e3:5.1f}")
        else:
            timing = "dt n/a"
        return (f"raw={raw_hz:6.1f}Hz fresh={fresh_hz:6.1f}Hz "
                f"dup={dup_hz:6.1f}Hz({dup_pct:4.0f}%) | {timing}")

    def reset(self):
        self.raw = self.fresh = self.dup = 0
        self.dts = []
        # keep last_boot / last_fresh_wall so cross-window freshness stays correct


def set_msg_interval(m, msg_id, hz):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            msg_id, int(1e6 / hz) if hz > 0 else 0, 0, 0, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser(description="Measure real ATTITUDE / LOCAL_POSITION_NED rate")
    ap.add_argument("--connect", default="/dev/ttyACM0",
                    help="serial device (real FC, e.g. /dev/ttyACM0) or tcp:host:port (SITL)")
    ap.add_argument("--baud", type=int, default=115200,
                    help="serial baud (USB is nominal; 115200 fine). Ignored for TCP/UDP.")
    ap.add_argument("--freq", type=float, default=100.0, help="requested stream rate [Hz]")
    ap.add_argument("--seconds", type=float, default=0.0, help="run time (0 = forever)")
    args = ap.parse_args()

    print(f"[i] Connecting to {args.connect} (baud {args.baud}) ...")
    m = mavutil.mavlink_connection(args.connect, baud=args.baud, source_system=255)
    m.wait_heartbeat()
    print(f"[i] Heartbeat sys {m.target_system}. Requesting ATTITUDE + "
          f"LOCAL_POSITION_NED at {args.freq:.0f} Hz.\n")

    set_msg_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, args.freq)
    set_msg_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, args.freq)

    stats = defaultdict(StreamStat)
    # latest values to print
    roll = pitch = yaw = 0.0
    x = y = z = vx = vy = vz = 0.0

    t_start = time.perf_counter()
    win_start = t_start
    try:
        while True:
            msg = m.recv_match(blocking=True, timeout=1.0)
            now = time.perf_counter()
            if msg is not None:
                t = msg.get_type()
                if t == "ATTITUDE":
                    roll, pitch, yaw = msg.roll, msg.pitch, msg.yaw
                    stats["ATTITUDE"].update(msg.time_boot_ms, now)
                elif t == "LOCAL_POSITION_NED":
                    x, y, z = msg.x, msg.y, msg.z
                    vx, vy, vz = msg.vx, msg.vy, msg.vz
                    stats["LOCAL_POSITION_NED"].update(msg.time_boot_ms, now)

            # 1 Hz report
            if now - win_start >= 1.0:
                window = now - win_start
                print(f"--- t={now - t_start:5.1f}s -------------------------------------------------")
                print(f"  ATTITUDE : {stats['ATTITUDE'].summary(window)}")
                print(f"             roll={math.degrees(roll):+6.1f} pitch={math.degrees(pitch):+6.1f} "
                      f"yaw={math.degrees(yaw):+6.1f} deg")
                print(f"  LOCAL_NED: {stats['LOCAL_POSITION_NED'].summary(window)}")
                print(f"             x={x:+6.2f} y={y:+6.2f} z={z:+6.2f} m  "
                      f"vx={vx:+5.2f} vy={vy:+5.2f} vz={vz:+5.2f} m/s  (alt={-z:+.2f})")
                for s in stats.values():
                    s.reset()
                win_start = now

            if args.seconds > 0 and (now - t_start) >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n[i] Stopped.")


if __name__ == "__main__":
    main()
