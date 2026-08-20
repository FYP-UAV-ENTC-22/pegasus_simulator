#!/usr/bin/env python3
"""
CTBR (Collective-Thrust + Body-Rate) hover controller over MAVLink.

Connects to ArduPilot SITL *through MAVProxy* (an extra UDP output), streams
ATTITUDE / LOCAL_POSITION_NED / GPS at 100 Hz, and closes a hover loop that sends
ONLY body rates + total thrust (SET_ATTITUDE_TARGET, attitude quaternion ignored).

Control structure (all we can do with CTBR):
  - Altitude hold : PID on altitude error (from LOCAL_POSITION_NED, alt = -z) -> thrust
  - Level attitude: P on roll/pitch angle                                    -> roll/pitch body rate
  - Heading hold  : P on yaw error                                           -> yaw body rate
Horizontal x/y drift is NOT corrected — that would need tilt/position control,
which CTBR alone cannot express. Target height defaults to 1.0 m.

The script also measures how fast it is actually RECEIVING each stream and
SENDING commands, and checks message freshness (time_boot_ms advancing), so you
can confirm the real 100 Hz rates.

Setup (SITL already running under Pegasus):
  In the MAVProxy console add a dedicated output for this script:
      output add 127.0.0.1:14561
  Then run (any python with pymavlink, e.g. venv-ardupilot or pegasus_env):
      python3 ctbr_hover_mavlink.py --connect udpin:127.0.0.1:14561
"""

import argparse
import math
import time
from collections import defaultdict

from pymavlink import mavutil

# ---- SET_ATTITUDE_TARGET type_mask bits (MAVLink ATTITUDE_TARGET_TYPEMASK) ----
TM_BODY_ROLL_RATE_IGNORE  = 1
TM_BODY_PITCH_RATE_IGNORE = 2
TM_BODY_YAW_RATE_IGNORE   = 4
TM_THRUST_BODY_SET        = 8
TM_THROTTLE_IGNORE        = 32
TM_ATTITUDE_IGNORE        = 64
# We USE the 3 body rates + throttle, and IGNORE the attitude quaternion:
CTBR_TYPE_MASK = TM_ATTITUDE_IGNORE  # = 64


def wrap_pi(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PID:
    """Simple PID with derivative-on-measurement and integral clamping."""
    def __init__(self, kp, ki, kd, i_limit, out_lo, out_hi):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.out_lo, self.out_hi = out_lo, out_hi
        self.integ = 0.0

    def reset(self):
        self.integ = 0.0

    def step(self, error, measurement_rate, dt, feedforward=0.0):
        # measurement_rate is d(measurement)/dt, used for the D term (no kick).
        self.integ = clamp(self.integ + error * dt, -self.i_limit, self.i_limit)
        out = feedforward + self.kp * error + self.ki * self.integ - self.kd * measurement_rate
        return clamp(out, self.out_lo, self.out_hi)


class RateMonitor:
    """Tracks receive/send rates and message freshness over 1 s windows."""
    def __init__(self):
        self.count = defaultdict(int)
        self.fresh = defaultdict(int)
        self.last_boot = {}
        self.t0 = time.perf_counter()

    def rx(self, name, time_boot_ms=None):
        self.count[name] += 1
        if time_boot_ms is not None:
            prev = self.last_boot.get(name)
            if prev is None or time_boot_ms > prev:
                self.fresh[name] += 1
            self.last_boot[name] = time_boot_ms

    def tx(self):
        self.count["TX_CMD"] += 1

    def report_and_reset(self):
        dt = time.perf_counter() - self.t0
        parts = []
        for name in sorted(self.count):
            hz = self.count[name] / dt
            if name in self.fresh:
                parts.append(f"{name}={hz:5.1f}Hz(fresh {self.fresh[name]/dt:5.1f})")
            else:
                parts.append(f"{name}={hz:5.1f}Hz")
        self.count.clear()
        self.fresh.clear()
        self.t0 = time.perf_counter()
        return " | ".join(parts)


class State:
    """Latest vehicle state from incoming MAVLink messages."""
    def __init__(self):
        self.roll = self.pitch = self.yaw = 0.0        # rad
        self.p = self.q = self.r = 0.0                 # body rates rad/s
        self.x = self.y = self.z = 0.0                 # NED position m
        self.vx = self.vy = self.vz = 0.0              # NED velocity m/s
        self.lat = self.lon = self.alt_msl = None      # GPS (deg, m)
        self.have_att = False
        self.have_pos = False
        self.armed = False                             # from HEARTBEAT
        self.custom_mode = None                        # from HEARTBEAT

    @property
    def altitude(self):
        """Height above EKF origin (positive up). NED z is down-positive."""
        return -self.z

    @property
    def climb_rate(self):
        """Positive when ascending."""
        return -self.vz


def set_message_interval(m, msg_id, rate_hz):
    """Request a message at rate_hz using MAV_CMD_SET_MESSAGE_INTERVAL."""
    interval_us = 0 if rate_hz <= 0 else int(1_000_000 / rate_hz)
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        msg_id, interval_us, 0, 0, 0, 0, 0)


def param_set(m, name, value, ptype=mavutil.mavlink.MAV_PARAM_TYPE_INT32):
    m.mav.param_set_send(m.target_system, m.target_component,
                         name.encode("ascii"), float(value), ptype)


def send_ctbr(m, roll_rate, pitch_rate, yaw_rate, thrust, t0):
    """Send one CTBR command: body rates (rad/s) + collective thrust (0..1).

    NOTE: target_component MUST be the autopilot (1). Addressing component 0
    causes ArduCopter to silently ignore SET_ATTITUDE_TARGET (verified).
    """
    m.mav.set_attitude_target_send(
        int((time.perf_counter() - t0) * 1000) & 0xFFFFFFFF,  # time_boot_ms
        m.target_system, 1,     # <-- component 1 (autopilot), not m.target_component
        CTBR_TYPE_MASK,
        [1.0, 0.0, 0.0, 0.0],   # quaternion (ignored)
        roll_rate, pitch_rate, yaw_rate,
        thrust)


def drain(m, st, mon):
    """Read all pending messages, update state, and count them for rate stats."""
    while True:
        msg = m.recv_match(blocking=False)
        if msg is None:
            return
        t = msg.get_type()
        if t == "HEARTBEAT":
            st.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            st.custom_mode = msg.custom_mode
        elif t == "ATTITUDE":
            st.roll, st.pitch, st.yaw = msg.roll, msg.pitch, msg.yaw
            st.p, st.q, st.r = msg.rollspeed, msg.pitchspeed, msg.yawspeed
            st.have_att = True
            mon.rx("ATTITUDE", msg.time_boot_ms)
        elif t == "LOCAL_POSITION_NED":
            st.x, st.y, st.z = msg.x, msg.y, msg.z
            st.vx, st.vy, st.vz = msg.vx, msg.vy, msg.vz
            st.have_pos = True
            mon.rx("LOCAL_POS", msg.time_boot_ms)
        elif t == "GLOBAL_POSITION_INT":
            st.lat, st.lon = msg.lat / 1e7, msg.lon / 1e7
            st.alt_msl = msg.alt / 1000.0
            mon.rx("GLOBAL_POS", msg.time_boot_ms)
        elif t == "GPS_RAW_INT":
            mon.rx("GPS_RAW")
        elif t == "STATUSTEXT":
            print(f"    [AP] {msg.text}")
        elif t == "COMMAND_ACK":
            res = mavutil.mavlink.enums["MAV_RESULT"].get(msg.result)
            res = res.name if res else msg.result
            print(f"    [ACK] cmd={msg.command} result={res}")


def get_param(m, name, timeout=3.0):
    """Read a parameter value (matched by id), or None on timeout."""
    m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
    t = time.time()
    while time.time() - t < timeout:
        p = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
        if p is not None and p.param_id.strip("\x00") == name:
            return p.param_value
    return None


def set_mode_confirm(m, st, mon, mode_name, timeout=5.0):
    """Set a flight mode and confirm via HEARTBEAT.custom_mode."""
    if mode_name not in m.mode_mapping():
        print(f"[!] Mode {mode_name} not available: {list(m.mode_mapping())}")
        return False
    target = m.mode_mapping()[mode_name]
    m.set_mode(target)
    t = time.time()
    while time.time() - t < timeout:
        drain(m, st, mon)
        if st.custom_mode == target:
            print(f"[i] Mode confirmed: {mode_name}")
            return True
        time.sleep(0.02)
    print(f"[!] Mode {mode_name} not confirmed (custom_mode={st.custom_mode})")
    return False


def main():
    ap = argparse.ArgumentParser(description="CTBR hover over MAVLink via MAVProxy")
    ap.add_argument("--connect", default="tcp:127.0.0.1:5762",
                    help="MAVLink endpoint. Prefer a DIRECT SITL port (tcp:127.0.0.1:5762 "
                         "or :5763) so SET_MESSAGE_INTERVAL is honored at 100 Hz. Going "
                         "through MAVProxy throttles streams to ~4 Hz.")
    ap.add_argument("--force-arm", action="store_true",
                    help="use the arm magic value to bypass arming checks")
    ap.add_argument("--alt", type=float, default=1.0, help="hover height [m]")
    ap.add_argument("--x", type=float, default=None,
                    help="NED north setpoint [m] (default: capture at arm)")
    ap.add_argument("--y", type=float, default=None,
                    help="NED east setpoint [m] (default: capture at arm)")
    ap.add_argument("--freq", type=float, default=100.0, help="control loop rate [Hz]")
    ap.add_argument("--hold", type=float, default=20.0, help="hold time at altitude [s]")
    ap.add_argument("--hover-thrust", type=float, default=0.5,
                    help="thrust feedforward guess (~hover), 0..1")
    args = ap.parse_args()

    # ------------------------------------------------------------------ connect
    print(f"[i] Connecting to {args.connect} ...")
    m = mavutil.mavlink_connection(args.connect, source_system=255)
    m.wait_heartbeat()
    print(f"[i] Heartbeat from system {m.target_system} component {m.target_component}")

    # ------------------------------------------------------------ request streams
    set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, args.freq)
    set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, args.freq)
    set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
    set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5)
    # GPS module output rate: 100 ms -> 10 Hz raw fixes (u-blox M8/M9/M10).
    param_set(m, "GPS_RATE_MS", 100)

    st = State()
    mon = RateMonitor()
    t0 = time.perf_counter()

    # Wait for first attitude + position so the controller starts with real data.
    print("[i] Waiting for ATTITUDE + LOCAL_POSITION_NED ...")
    while not (st.have_att and st.have_pos):
        drain(m, st, mon)
        time.sleep(0.005)
    print(f"[i] Got state. altitude={st.altitude:+.2f} m  yaw={math.degrees(st.yaw):+.1f} deg")
    yaw_setpoint = st.yaw  # hold current heading

    # ---------------------------------------------------------------- controllers
    # Altitude PID -> thrust (0..1). Feedforward = hover thrust; integrator finds
    # the true hover point. D term uses climb rate to damp.
    alt_pid = PID(kp=0.18, ki=0.06, kd=0.22, i_limit=0.35,
                  out_lo=0.10, out_hi=0.90)
    # Attitude leveling: angle (rad) -> body rate (rad/s), simple P (ArduPilot's
    # inner rate controller does the damping).
    KP_ATT = 6.0
    RATE_LIMIT = 2.0          # rad/s clamp on roll/pitch rate commands
    KP_YAW = 2.5
    YAW_RATE_LIMIT = 1.5

    # Outer position loop (NED x/y) -> desired horizontal accel -> tilt reference.
    KP_POS = 0.5              # position error [m] -> accel [m/s^2]
    KD_POS = 1.0             # velocity damping [m/s] -> accel [m/s^2]
    TILT_MAX = math.radians(12.0)   # cap the commanded roll/pitch reference
    G = 9.81
    # Integrator on x/y accel to trim steady wind/bias (small, clamped).
    xy_integ_n = 0.0
    xy_integ_e = 0.0
    KI_POS = 0.15
    XY_I_LIMIT = 1.5          # m/s^2

    # --------------------------------------------- param for clean CTBR control
    # GUID_OPTIONS bit 3 (value 8): use SET_ATTITUDE_TARGET thrust field directly.
    cur = get_param(m, "GUID_OPTIONS")
    if cur is not None and not (int(cur) & 8):
        param_set(m, "GUID_OPTIONS", int(cur) | 8)
        print(f"[i] GUID_OPTIONS {int(cur)} -> {int(cur) | 8}")

    # ------------------------------------ takeoff (native) then hand over to CTBR
    # ArduCopter will NOT spool motors from SET_ATTITUDE_TARGET while on the ground
    # (ground-idle protection). So take off with the native GUIDED controller to get
    # airborne, THEN switch to GUIDED_NOGPS and hold with pure CTBR commands.
    if not set_mode_confirm(m, st, mon, "GUIDED"):
        return

    print("[i] Arming ...")
    arm_p2 = 21196 if args.force_arm else 0   # magic value bypasses arming checks
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                            1, arm_p2, 0, 0, 0, 0, 0)
    t_arm = time.time()
    while time.time() - t_arm < 8.0 and not st.armed:
        drain(m, st, mon)            # surfaces [AP] reasons; updates st.armed
        time.sleep(0.02)
    if not st.armed:
        print("[!] NOT ARMED — see the [AP] reason above. Retry with --force-arm.")
        return
    print("[i] Armed.")

    ground_alt = st.altitude
    print(f"[i] Native takeoff to {args.alt:.2f} m ...")
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                            0, 0, 0, 0, 0, 0, args.alt)
    t_to = time.time()
    while time.time() - t_to < 15.0:
        drain(m, st, mon)
        if st.altitude - ground_alt >= 0.9 * args.alt:
            break
        time.sleep(0.02)
    print(f"[i] Airborne at {st.altitude - ground_alt:+.2f} m. Handing over to CTBR.")

    # ---- capture the x/y hold point (NED), now airborne ----
    x_sp = args.x if args.x is not None else st.x
    y_sp = args.y if args.y is not None else st.y

    # ---- switch to GUIDED_NOGPS: pure attitude/rate + thrust control ----
    if not set_mode_confirm(m, st, mon, "GUIDED_NOGPS"):
        print("[!] Could not enter GUIDED_NOGPS — landing.")
        set_mode_confirm(m, st, mon, "LAND")
        return
    print(f"[i] CTBR hold: NED x={x_sp:+.2f} y={y_sp:+.2f} alt={args.alt:.2f} m "
          f"(only body rates + thrust from here).")

    # -------------------------------------------------------------- control loop
    T = 1.0 / args.freq
    t_start = time.perf_counter()
    next_t = time.perf_counter()
    last_report = time.perf_counter()
    last_loop = time.perf_counter()
    jitter_max = 0.0

    phase = "CLIMB_HOLD"      # -> LAND -> DONE
    land_started = None

    print(f"[i] Hovering at {args.alt:.2f} m for {args.hold:.0f} s "
          f"(CTBR only). Ctrl-C to land early.\n")
    try:
        while True:
            now = time.perf_counter()
            dt = now - last_loop
            last_loop = now
            if dt <= 0:
                dt = T

            drain(m, st, mon)

            # ---- hold for the requested time, then land via ArduPilot LAND mode ----
            alt_sp = args.alt
            if now - t_start > args.hold:
                break

            # ---- altitude PID -> thrust ----
            alt_err = alt_sp - st.altitude
            thrust = alt_pid.step(alt_err, measurement_rate=st.climb_rate,
                                  dt=dt, feedforward=args.hover_thrust)

            # ---- outer position loop (NED) -> desired horizontal accel ----
            # PD on position/velocity, plus a small clamped integrator for bias.
            xy_integ_n = clamp(xy_integ_n + (x_sp - st.x) * dt, -XY_I_LIMIT / max(KI_POS, 1e-6), XY_I_LIMIT / max(KI_POS, 1e-6))
            xy_integ_e = clamp(xy_integ_e + (y_sp - st.y) * dt, -XY_I_LIMIT / max(KI_POS, 1e-6), XY_I_LIMIT / max(KI_POS, 1e-6))
            a_north = KP_POS * (x_sp - st.x) - KD_POS * st.vx + KI_POS * xy_integ_n
            a_east  = KP_POS * (y_sp - st.y) - KD_POS * st.vy + KI_POS * xy_integ_e

            # rotate world NED accel into the drone's heading frame (yaw = st.yaw)
            cy, sy = math.cos(st.yaw), math.sin(st.yaw)
            a_fwd   =  a_north * cy + a_east * sy      # along the nose
            a_right = -a_north * sy + a_east * cy      # out the right side

            # desired tilt from accel (small-angle: a ~= g*tilt), ArduPilot signs:
            #   roll +  -> accel right   ;   pitch + -> accel backward
            roll_ref  = clamp(+a_right / G, -TILT_MAX, TILT_MAX)
            pitch_ref = clamp(-a_fwd  / G, -TILT_MAX, TILT_MAX)

            # ---- inner attitude loop: angle error -> body rates ----
            roll_rate  = clamp(KP_ATT * (roll_ref  - st.roll),  -RATE_LIMIT, RATE_LIMIT)
            pitch_rate = clamp(KP_ATT * (pitch_ref - st.pitch), -RATE_LIMIT, RATE_LIMIT)
            yaw_rate   = clamp(KP_YAW * wrap_pi(yaw_setpoint - st.yaw),
                               -YAW_RATE_LIMIT, YAW_RATE_LIMIT)

            send_ctbr(m, roll_rate, pitch_rate, yaw_rate, thrust, t0)
            mon.tx()

            # ---- 1 Hz status: real RX/TX rates + control state ----
            loop_jit = abs((now - (next_t - T)))
            jitter_max = max(jitter_max, loop_jit)
            if now - last_report >= 1.0:
                dxy = math.hypot(x_sp - st.x, y_sp - st.y)
                print(f"[{phase:10s}] alt={st.altitude:+.2f}(sp{alt_sp:+.2f}) "
                      f"dxy={dxy:4.2f}m thr={thrust:.2f} "
                      f"r={math.degrees(st.roll):+5.1f} p={math.degrees(st.pitch):+5.1f} "
                      f"(ref {math.degrees(roll_ref):+4.1f}/{math.degrees(pitch_ref):+4.1f}) "
                      f"| {mon.report_and_reset()} | jit<={jitter_max*1e3:4.1f}ms")
                last_report = now
                jitter_max = 0.0

            # ---- precise fixed-rate scheduling ----
            next_t += T
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()  # fell behind; resync

    except KeyboardInterrupt:
        print("\n[i] Interrupted.")

    finally:
        # Land with ArduPilot's LAND mode (reliable touchdown + auto-disarm), rather
        # than trying to land on CTBR. This leaves the ground-gating to ArduPilot.
        print("[i] Switching to LAND ...")
        set_mode_confirm(m, st, mon, "LAND")
        t_land = time.time()
        while time.time() - t_land < 20.0 and st.armed:
            drain(m, st, mon)
            time.sleep(0.1)
        print("[i] Done." if not st.armed else "[i] Done (still armed — check vehicle).")


if __name__ == "__main__":
    main()
