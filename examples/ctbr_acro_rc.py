#!/usr/bin/env python3
"""
CTBR hover in ACRO via RC_CHANNELS_OVERRIDE — full control INCLUDING takeoff.

ArduCopter's ACRO mode is a pure body-rate + manual-collective-throttle mode, and it
takes off from the ground on throttle-up (no GUIDED land/spool lock). Driving its RC
channels over MAVLink therefore gives a genuine CTBR interface that can lift off:

    ch(roll)  -> roll  body rate       ch(throttle) -> collective thrust
    ch(pitch) -> pitch body rate        ch(yaw)      -> yaw   body rate

RC override sends PWM (1000..2000), so we invert ArduCopter's own stick->command maths
using parameters READ FROM THE VEHICLE:
    rate:      rate_degs = ACRO_RP_RATE * norm_input      (ACRO_*_EXPO forced to 0)
               norm_input -> PWM via RCx_MIN/MAX/TRIM      (RCx_DZ forced to 0)
    throttle:  cubic expo around THR_MID / MOT_THST_HOVER  (inverted numerically)
    ACRO_TRAINER forced to 0 so ACRO is pure rate (no auto-leveling to fight us).

Control (all done here; ArduPilot only runs the inner rate loop):
    altitude PID          -> collective thrust
    x/y position PID -> tilt ref -> attitude P -> roll/pitch body rate
    yaw hold P            -> yaw body rate

Setup: connect DIRECTLY to SITL (not through MAVProxy) so streams hit 100 Hz:
    python3 ctbr_acro_rc.py --connect tcp:127.0.0.1:5762 --alt 1
"""

import argparse
import math
import time
from collections import defaultdict

from pymavlink import mavutil


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PID:
    def __init__(self, kp, ki, kd, i_limit, out_lo, out_hi):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit, self.out_lo, self.out_hi = i_limit, out_lo, out_hi
        self.integ = 0.0

    def step(self, error, meas_rate, dt, ff=0.0):
        self.integ = clamp(self.integ + error * dt, -self.i_limit, self.i_limit)
        out = ff + self.kp * error + self.ki * self.integ - self.kd * meas_rate
        return clamp(out, self.out_lo, self.out_hi)


class RateMonitor:
    def __init__(self):
        self.count = defaultdict(int)
        self.fresh = defaultdict(int)
        self.last_boot = {}
        self.t0 = time.perf_counter()

    def rx(self, name, boot=None):
        self.count[name] += 1
        if boot is not None:
            prev = self.last_boot.get(name)
            if prev is None or boot > prev:
                self.fresh[name] += 1
            self.last_boot[name] = boot

    def tx(self):
        self.count["TX"] += 1

    def report(self):
        dt = time.perf_counter() - self.t0
        parts = []
        for n in sorted(self.count):
            hz = self.count[n] / dt
            parts.append(f"{n}={hz:5.1f}" + (f"(f{self.fresh[n]/dt:4.1f})" if n in self.fresh else ""))
        self.count.clear(); self.fresh.clear(); self.t0 = time.perf_counter()
        return " ".join(parts)


class State:
    def __init__(self):
        self.roll = self.pitch = self.yaw = 0.0
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.armed = False
        self.custom_mode = None
        self.have_att = self.have_pos = False

    @property
    def altitude(self):
        return -self.z

    @property
    def climb_rate(self):
        return -self.vz


class Chan:
    """One RC channel's calibration, used to invert norm_input -> PWM."""
    def __init__(self, mn, mx, tr):
        self.min, self.max, self.trim = int(mn), int(mx), int(tr)

    def pwm_from_norm(self, n):
        """norm_input in [-1,1] (deadzone forced to 0) -> PWM."""
        n = clamp(n, -1.0, 1.0)
        if n >= 0:
            pwm = self.trim + n * (self.max - self.trim)
        else:
            pwm = self.trim + n * (self.trim - self.min)
        return int(clamp(pwm, self.min, self.max))


# --------------------------------------------------------------- param helpers
def get_param(m, name, timeout=3.0):
    m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
    t = time.time()
    while time.time() - t < timeout:
        p = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
        if p is not None and p.param_id.strip("\x00") == name:
            return p.param_value
    return None


def set_param(m, name, value, ptype=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(value), ptype)
    time.sleep(0.05)


def set_msg_interval(m, msg_id, hz):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            msg_id, int(1e6 / hz) if hz > 0 else 0, 0, 0, 0, 0, 0)


def set_mode_confirm(m, st, mon, name, timeout=5.0):
    if name not in m.mode_mapping():
        print(f"[!] mode {name} unavailable"); return False
    target = m.mode_mapping()[name]
    m.set_mode(target)
    t = time.time()
    while time.time() - t < timeout:
        drain(m, st, mon)
        if st.custom_mode == target:
            print(f"[i] Mode confirmed: {name}"); return True
        time.sleep(0.02)
    print(f"[!] mode {name} not confirmed (is {st.custom_mode})"); return False


# ------------------------------------------------------------- throttle mapping
def throttle_forward(throttle_control, thr_mid, mid_stick):
    """ArduCopter get_pilot_desired_throttle: 0..1000 stick -> 0..1 throttle."""
    tc = clamp(throttle_control, 0, 1000)
    if tc < mid_stick:
        thr_in = tc * 0.5 / mid_stick
    else:
        thr_in = 0.5 + (tc - mid_stick) * 0.5 / (1000 - mid_stick)
    expo = clamp(-(thr_mid - 0.5) / 0.375, -0.5, 1.0)
    return thr_in * (1.0 - expo) + expo * thr_in ** 3


def thrust_to_pwm(thrust, thr_mid, mid_stick, ch3):
    """Invert throttle_forward numerically, then map 0..1000 -> PWM on ch3."""
    thrust = clamp(thrust, 0.0, 1.0)
    lo, hi = 0.0, 1000.0
    for _ in range(30):                      # binary search (monotonic)
        mid = 0.5 * (lo + hi)
        if throttle_forward(mid, thr_mid, mid_stick) < thrust:
            lo = mid
        else:
            hi = mid
    tc = 0.5 * (lo + hi)
    pwm = ch3.min + (tc / 1000.0) * (ch3.max - ch3.min)
    return int(clamp(pwm, ch3.min, ch3.max))


# --------------------------------------------------------------------- drain
def drain(m, st, mon):
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
            st.have_att = True
            mon.rx("ATT", msg.time_boot_ms)
        elif t == "LOCAL_POSITION_NED":
            st.x, st.y, st.z = msg.x, msg.y, msg.z
            st.vx, st.vy, st.vz = msg.vx, msg.vy, msg.vz
            st.have_pos = True
            mon.rx("POS", msg.time_boot_ms)
        elif t == "STATUSTEXT":
            print(f"    [AP] {msg.text}")
        elif t == "COMMAND_ACK":
            r = mavutil.mavlink.enums["MAV_RESULT"].get(msg.result)
            print(f"    [ACK] cmd={msg.command} {r.name if r else msg.result}")


def send_rc(m, ch_roll, ch_pitch, ch_thr, ch_yaw, pwm_map, mon):
    """Send RC_CHANNELS_OVERRIDE. pwm_map: dict channel_number(1..8) -> pwm.
    Channels not in pwm_map are released (0)."""
    vals = [pwm_map.get(i, 0) for i in range(1, 9)]     # 8 base channels
    m.mav.rc_channels_override_send(m.target_system, 1, *vals)
    mon.tx()


def main():
    ap = argparse.ArgumentParser(description="CTBR hover in ACRO via RC override (takeoff included)")
    ap.add_argument("--connect", default="tcp:127.0.0.1:5762")
    ap.add_argument("--alt", type=float, default=1.0, help="hover height [m]")
    ap.add_argument("--freq", type=float, default=100.0)
    ap.add_argument("--hold", type=float, default=20.0)
    ap.add_argument("--force-arm", action="store_true")
    args = ap.parse_args()

    print(f"[i] Connecting to {args.connect} ...")
    m = mavutil.mavlink_connection(args.connect, source_system=255)
    m.wait_heartbeat()
    print(f"[i] Heartbeat sys {m.target_system}")

    # ---- force clean ACRO behaviour, then read the calibration we invert ----
    set_param(m, "ACRO_TRAINER", 0)        # pure rate, no auto-level
    set_param(m, "ACRO_RP_EXPO", 0)        # linear rate mapping
    set_param(m, "ACRO_Y_EXPO", 0)
    # channel map (which RC channel is roll/pitch/thr/yaw); default 1/2/3/4
    cmap = {k: int(get_param(m, f"RCMAP_{k}") or d)
            for k, d in (("ROLL", 1), ("PITCH", 2), ("THROTTLE", 3), ("YAW", 4))}
    for k in ("ROLL", "PITCH", "YAW"):
        set_param(m, f"RC{cmap[k]}_DZ", 0)   # kill deadzone on rate channels
    def chan(ch):
        return Chan(get_param(m, f"RC{ch}_MIN") or 1000,
                    get_param(m, f"RC{ch}_MAX") or 2000,
                    get_param(m, f"RC{ch}_TRIM") or 1500)
    ch_roll, ch_pitch, ch_thr, ch_yaw = (chan(cmap["ROLL"]), chan(cmap["PITCH"]),
                                         chan(cmap["THROTTLE"]), chan(cmap["YAW"]))
    acro_rp_rate = float(get_param(m, "ACRO_RP_RATE") or 360.0)   # deg/s at full stick
    acro_y_rate = float(get_param(m, "ACRO_Y_RATE") or 202.5)
    thr_mid = float(get_param(m, "MOT_THST_HOVER") or 0.35)       # hover throttle 0..1
    mid_stick = float(get_param(m, "THR_MID") or 500.0)           # mid-stick 0..1000
    print(f"[i] ACRO_RP_RATE={acro_rp_rate:.0f} deg/s  ACRO_Y_RATE={acro_y_rate:.0f}  "
          f"MOT_THST_HOVER={thr_mid:.2f}  THR_MID={mid_stick:.0f}")
    print(f"[i] RCMAP r/p/t/y={cmap['ROLL']}/{cmap['PITCH']}/{cmap['THROTTLE']}/{cmap['YAW']}  "
          f"thr ch min/trim/max={ch_thr.min}/{ch_thr.trim}/{ch_thr.max}")

    def rate_to_pwm(rate_rads, rate_max_degs, ch):
        n = clamp(math.degrees(rate_rads) / rate_max_degs, -1.0, 1.0)
        return ch.pwm_from_norm(n)

    def build_pwm(roll_rate, pitch_rate, yaw_rate, thrust):
        return {
            cmap["ROLL"]:     rate_to_pwm(roll_rate, acro_rp_rate, ch_roll),
            cmap["PITCH"]:    rate_to_pwm(pitch_rate, acro_rp_rate, ch_pitch),
            cmap["YAW"]:      rate_to_pwm(yaw_rate, acro_y_rate, ch_yaw),
            cmap["THROTTLE"]: thrust_to_pwm(thrust, thr_mid, mid_stick, ch_thr),
        }

    # ---- streams ----
    set_msg_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, args.freq)
    set_msg_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, args.freq)

    st = State(); mon = RateMonitor()
    print("[i] Waiting for ATTITUDE + LOCAL_POSITION_NED ...")
    while not (st.have_att and st.have_pos):
        drain(m, st, mon); time.sleep(0.005)
    yaw_sp = st.yaw
    x_sp, y_sp = st.x, st.y
    print(f"[i] State ok. alt={st.altitude:+.2f} yaw={math.degrees(st.yaw):+.1f} deg")

    # ---- controllers ----
    alt_pid = PID(kp=0.35, ki=0.15, kd=0.30, i_limit=0.4, out_lo=0.0, out_hi=0.95)
    KP_ATT = 6.0; RATE_LIM = 3.0
    KP_YAW = 2.5; YAW_LIM = 1.5
    KP_POS = 0.5; KD_POS = 1.0; KI_POS = 0.15; TILT_MAX = math.radians(12.0); G = 9.81
    xin = xie = 0.0
    XY_I = 1.5

    # ---- ACRO + arm (throttle at min so the arming check passes) ----
    if not set_mode_confirm(m, st, mon, "ACRO"):
        return
    thr_min_pwm = ch_thr.min
    idle = {cmap["ROLL"]: ch_roll.trim, cmap["PITCH"]: ch_pitch.trim,
            cmap["YAW"]: ch_yaw.trim, cmap["THROTTLE"]: thr_min_pwm}
    for _ in range(30):
        send_rc(m, ch_roll, ch_pitch, ch_thr, ch_yaw, idle, mon)
        drain(m, st, mon); time.sleep(0.01)

    print("[i] Arming (ACRO, throttle min) ...")
    arm_p2 = 21196 if args.force_arm else 0
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                            1, arm_p2, 0, 0, 0, 0, 0)
    t_arm = time.time()
    while time.time() - t_arm < 8.0 and not st.armed:
        send_rc(m, ch_roll, ch_pitch, ch_thr, ch_yaw, idle, mon)
        drain(m, st, mon); time.sleep(0.01)
    if not st.armed:
        print("[!] NOT ARMED — see [AP]. Try --force-arm."); return
    print("[i] Armed. Taking off with CTBR (throttle-up) ...")

    ground_alt = st.altitude
    T = 1.0 / args.freq
    t0 = time.perf_counter(); t_start = t0
    next_t = t0; last_report = t0; last_loop = t0; jit = 0.0

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_loop; last_loop = now
            if dt <= 0:
                dt = T
            drain(m, st, mon)

            # altitude setpoint: ramp up from ground over 2 s, then hold
            climb_t = now - t_start
            alt_sp = ground_alt + min(args.alt, 0.5 * climb_t) if climb_t < 2.0 else ground_alt + args.alt
            if climb_t > args.hold:
                break

            # altitude PID -> thrust (feedforward = hover throttle)
            thrust = alt_pid.step(alt_sp - st.altitude, st.climb_rate, dt, ff=thr_mid)

            # position PD(+I) -> desired accel -> tilt ref
            xin = clamp(xin + (x_sp - st.x) * dt, -XY_I / max(KI_POS, 1e-6), XY_I / max(KI_POS, 1e-6))
            xie = clamp(xie + (y_sp - st.y) * dt, -XY_I / max(KI_POS, 1e-6), XY_I / max(KI_POS, 1e-6))
            a_n = KP_POS * (x_sp - st.x) - KD_POS * st.vx + KI_POS * xin
            a_e = KP_POS * (y_sp - st.y) - KD_POS * st.vy + KI_POS * xie
            cy, sy = math.cos(st.yaw), math.sin(st.yaw)
            a_fwd = a_n * cy + a_e * sy
            a_rgt = -a_n * sy + a_e * cy
            roll_ref = clamp(a_rgt / G, -TILT_MAX, TILT_MAX)
            pitch_ref = clamp(-a_fwd / G, -TILT_MAX, TILT_MAX)

            # attitude P -> body rates
            roll_rate = clamp(KP_ATT * (roll_ref - st.roll), -RATE_LIM, RATE_LIM)
            pitch_rate = clamp(KP_ATT * (pitch_ref - st.pitch), -RATE_LIM, RATE_LIM)
            yaw_rate = clamp(KP_YAW * wrap_pi(yaw_sp - st.yaw), -YAW_LIM, YAW_LIM)

            send_rc(m, ch_roll, ch_pitch, ch_thr, ch_yaw,
                    build_pwm(roll_rate, pitch_rate, yaw_rate, thrust), mon)

            jit = max(jit, abs(now - (next_t - T)))
            if now - last_report >= 1.0:
                dxy = math.hypot(x_sp - st.x, y_sp - st.y)
                pwm = build_pwm(roll_rate, pitch_rate, yaw_rate, thrust)
                print(f"alt={st.altitude:+.2f}(sp{alt_sp:+.2f}) dxy={dxy:4.2f} thr={thrust:.2f} "
                      f"thrPWM={pwm[cmap['THROTTLE']]} r={math.degrees(st.roll):+5.1f} "
                      f"p={math.degrees(st.pitch):+5.1f} | {mon.report()} | jit<={jit*1e3:4.1f}ms")
                last_report = now; jit = 0.0

            next_t += T
            s = next_t - time.perf_counter()
            if s > 0:
                time.sleep(s)
            else:
                next_t = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[i] Interrupted.")
    finally:
        print("[i] Landing (LAND mode) ...")
        # release override so LAND controls freely
        m.mav.rc_channels_override_send(m.target_system, 1, *([0] * 8))
        set_mode_confirm(m, st, mon, "LAND")
        t_l = time.time()
        while time.time() - t_l < 20.0 and st.armed:
            drain(m, st, mon); time.sleep(0.1)
        print("[i] Done." if not st.armed else "[i] Done (still armed).")


if __name__ == "__main__":
    main()
