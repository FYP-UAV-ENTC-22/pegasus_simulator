#!/usr/bin/env python3
"""
Live roll & pitch plot from the flight controller, with measured frequency.

- A background thread drains MAVLink at full speed (so nothing is missed) and
  requests ATTITUDE at --freq (default 100 Hz).
- Two live graphs (roll, pitch, in degrees) scroll in real time.
- The window title shows the MEASURED rate: raw Hz and fresh Hz (fresh = unique
  time_boot_ms; raw-fresh gap = duplicates), so you see the true 100 Hz.

The plot itself redraws at ~30 fps (a GUI can't usefully redraw at 100 Hz), but
every 100 Hz sample is captured in the buffer and shown.

Usage (real FC):   python3 live_roll_pitch.py --connect /dev/ttyACM0 --freq 100
       (SITL):     python3 live_roll_pitch.py --connect tcp:127.0.0.1:5762
"""

import argparse
import math
import threading
import time
from collections import deque

import matplotlib
matplotlib.use("GTK3Agg", force=True)          # the toolkit available on this machine
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pymavlink import mavutil


class Reader(threading.Thread):
    """Background MAVLink reader: fills buffers + measures rate. Never commands."""
    def __init__(self, conn, baud, freq, window_s):
        super().__init__(daemon=True)
        self.conn, self.baud, self.freq = conn, baud, freq
        self.maxlen = int(window_s * max(freq, 50) * 1.5)
        self.lock = threading.Lock()
        self.t = deque(maxlen=self.maxlen)
        self.roll = deque(maxlen=self.maxlen)
        self.pitch = deque(maxlen=self.maxlen)
        self.raw_hz = 0.0
        self.fresh_hz = 0.0
        self.running = True
        self.t0 = None

    def run(self):
        m = mavutil.mavlink_connection(self.conn, baud=self.baud, source_system=255)
        m.wait_heartbeat()
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
                                int(1e6 / self.freq), 0, 0, 0, 0, 0)
        self.t0 = time.perf_counter()
        last_boot = None
        raw = fresh = 0
        win = self.t0
        while self.running:
            msg = m.recv_match(type="ATTITUDE", blocking=True, timeout=0.5)
            now = time.perf_counter()
            if msg is not None:
                raw += 1
                if last_boot is None or msg.time_boot_ms > last_boot:
                    fresh += 1
                    last_boot = msg.time_boot_ms
                with self.lock:
                    self.t.append(now - self.t0)
                    self.roll.append(math.degrees(msg.roll))
                    self.pitch.append(math.degrees(msg.pitch))
            if now - win >= 0.5:                 # update rate estimate twice a second
                dt = now - win
                with self.lock:
                    self.raw_hz = raw / dt
                    self.fresh_hz = fresh / dt
                raw = fresh = 0
                win = now

    def snapshot(self):
        with self.lock:
            return (list(self.t), list(self.roll), list(self.pitch),
                    self.raw_hz, self.fresh_hz)


def main():
    ap = argparse.ArgumentParser(description="Live roll/pitch plot + measured frequency")
    ap.add_argument("--connect", default="/dev/ttyACM0",
                    help="serial device (/dev/ttyACM0) or tcp:host:port (SITL)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--freq", type=float, default=100.0, help="requested ATTITUDE rate [Hz]")
    ap.add_argument("--window", type=float, default=10.0, help="seconds of history shown")
    args = ap.parse_args()

    print(f"[i] Connecting to {args.connect} ... requesting ATTITUDE at {args.freq:.0f} Hz")
    reader = Reader(args.connect, args.baud, args.freq, args.window)
    reader.start()

    # wait for first samples
    t_wait = time.time()
    while not reader.t and time.time() - t_wait < 8:
        time.sleep(0.05)
    if not reader.t:
        print("[!] No ATTITUDE received — check connection/port."); return

    # ---- figure: two stacked graphs ----
    fig, (ax_r, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.canvas.manager.set_window_title("Roll / Pitch live")
    (line_r,) = ax_r.plot([], [], color="#e5484d", lw=1.5)
    (line_p,) = ax_p.plot([], [], color="#3b82f6", lw=1.5)
    ax_r.set_ylabel("roll [deg]");  ax_r.grid(True, alpha=0.3)
    ax_p.set_ylabel("pitch [deg]"); ax_p.grid(True, alpha=0.3)
    ax_p.set_xlabel("time [s]")
    val_r = ax_r.text(0.99, 0.92, "", transform=ax_r.transAxes, ha="right", va="top",
                      fontfamily="monospace")
    val_p = ax_p.text(0.99, 0.92, "", transform=ax_p.transAxes, ha="right", va="top",
                      fontfamily="monospace")

    def update(_frame):
        t, roll, pitch, raw_hz, fresh_hz = reader.snapshot()
        if not t:
            return line_r, line_p
        line_r.set_data(t, roll)
        line_p.set_data(t, pitch)
        tmax = t[-1]
        for ax in (ax_r, ax_p):
            ax.set_xlim(max(0, tmax - args.window), tmax + 0.1)
        for ax, data in ((ax_r, roll), (ax_p, pitch)):
            lo, hi = min(data), max(data)
            pad = max(1.0, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        dup_pct = 100.0 * (raw_hz - fresh_hz) / raw_hz if raw_hz > 0 else 0.0
        fig.suptitle(f"raw {raw_hz:5.1f} Hz   fresh {fresh_hz:5.1f} Hz   "
                     f"dup {dup_pct:3.0f}%   (requested {args.freq:.0f} Hz)",
                     fontfamily="monospace")
        val_r.set_text(f"{roll[-1]:+6.1f}°")
        val_p.set_text(f"{pitch[-1]:+6.1f}°")
        return line_r, line_p

    # redraw ~30 fps; the reader thread keeps capturing at full rate underneath
    ani = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    print("[i] Plotting. Close the window (or Ctrl-C) to stop.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        reader.running = False


if __name__ == "__main__":
    main()
