#!/usr/bin/env python3
"""Guarded relative teleoperation, optionally with the so101_record dataset writer."""
import argparse
import json
import math
import os
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import termios
import time
import tty

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from so101_record_ui import CommandInput

TICKS_PER_DEGREE = 4095 / 360
START_TOLERANCE_TICKS = 8
PROFILES = {
    "gentle": {"gain": 0.3, "travel_deg": 15, "speed_deg_s": 8, "leader_limit_deg": 100},
    "expanded": {"gain": 1.0, "travel_deg": 30, "speed_deg_s": 15, "leader_limit_deg": 150},
    "wide": {"gain": 1.0, "travel_deg": 60, "speed_deg_s": 15, "leader_limit_deg": 150},
    "full": {"gain": 1.0, "travel_deg": None, "speed_deg_s": 30, "leader_limit_deg": None},
}
LABELS = {"shoulder_pan": "底座转向", "shoulder_lift": "肩部俯仰", "elbow_flex": "肘部弯曲",
          "wrist_flex": "手腕俯仰", "wrist_roll": "手腕旋转", "gripper": "夹爪开合"}


def enable_verified_torque(bus, motors):
    """Retry transient serial replies, then confirm each requested motor is on."""
    names = [motors] if isinstance(motors, str) else list(motors)
    # Feetech enables each motor with two acknowledged writes (Torque_Enable,
    # Lock). A corrupted reply to either write is safe to retry: both are
    # idempotent, and Goal_Position was already seeded before this call.
    bus.enable_torque(names, num_retry=5)
    status = bus.sync_read('Torque_Enable', names, normalize=False, num_retry=3)
    disabled = [name for name in names if status.get(name) != 1]
    if disabled:
        raise RuntimeError(f'力矩启动后复核失败：{disabled}。已停止并尝试关闭力矩。')


def boundary(current, low, high):
    """Keep the initial goal unchanged, with inward-only motion near endpoints."""
    if not 0 <= low < high <= 4095:
        return None, "校准范围无效，需重新检查校准"
    hard_low, hard_high = max(35, low), min(4060, high)
    if not hard_low - START_TOLERANCE_TICKS <= current <= hard_high + START_TOLERANCE_TICKS:
        direction = "增大" if current < hard_low else "减小"
        target = min(hard_high, hard_low + 35) if current < hard_low else max(hard_low, hard_high - 35)
        return None, f"禁止启动：读数需{direction}到约{target}；超出校准范围或靠近编码器零点"
    soft_low, soft_high = max(hard_low, low + 15), min(hard_high, high - 15)
    if soft_low >= soft_high:
        return None, "校准范围过窄"
    # Small measured overshoot may be accepted, but every commanded goal stays inside.
    bounds = min(soft_low, max(hard_low, current)), max(soft_high, min(hard_high, current))
    if current < hard_low or current > hard_high:
        return bounds, f"可启动：超出校准边界{hard_low-current if current < hard_low else current-hard_high}刻度；目标会限制在边界内"
    if current < soft_low:
        return bounds, "可启动：靠近下限，只允许向读数增大方向离开起点"
    if current > soft_high:
        return bounds, "可启动：靠近上限，只允许向读数减小方向离开起点"
    return bounds, "可启动"


def report(follower, positions, names):
    bounds = {}
    print("\n白臂启动检查（编码器刻度；增减方向以屏幕读数为准）：")
    for name in names:
        c = follower.calibration[name]
        b, reason = boundary(positions[name], c.range_min, c.range_max)
        print(f"{LABELS[name]} ({name}): 当前 {positions[name]}, 校准 [{c.range_min}, {c.range_max}] | {reason}")
        if b is not None:
            bounds[name] = b
    return bounds if len(bounds) == len(names) else None


def next_target(master, master_zero, follower_zero, previous, actual, low, high, dt,
                gain=0.3, travel_deg=15, speed_deg_s=8):
    # Use a fixed initial anchor: drift cannot accumulate into unlimited travel.
    # Full mode receives an unwrapped leader position from Alignment.
    delta = (master - master_zero) if travel_deg is None else (master - master_zero + 2048) % 4096 - 2048
    if travel_deg is None:
        desired = follower_zero + gain * delta
    else:
        travel = travel_deg * TICKS_PER_DEGREE
        desired = follower_zero + max(-travel, min(travel, gain * delta))
    desired = max(low, min(high, desired))
    step = speed_deg_s * TICKS_PER_DEGREE * min(dt, 0.05)
    goal = max(previous - step, min(previous + step, desired))
    # Abort on lag instead of allowing the target to run away from a stuck joint.
    if abs(previous - actual) > 8 * TICKS_PER_DEGREE:
        raise RuntimeError("从臂位置偏离目标超过8°，停止测试")
    return max(low, min(high, goal))


def diagnostic(master, master_zero, follower_zero, previous, actual, low, high, goal,
               gain=0.3, travel_deg=15, session_low=None, session_high=None):
    delta = (master - master_zero) if travel_deg is None else (master - master_zero + 2048) % 4096 - 2048
    requested = follower_zero + gain * delta
    if travel_deg is None:
        travel_goal = requested
    else:
        travel = travel_deg * TICKS_PER_DEGREE
        travel_goal = max(follower_zero - travel, min(follower_zero + travel, requested))
    bounded = max(low, min(high, travel_goal))
    session_goal = bounded if session_low is None else max(session_low, min(session_high, bounded))
    return {
        "master_raw": master, "actual_raw": actual,
        "previous_target_raw": previous, "command_raw": round(goal),
        "master_delta_deg": delta / TICKS_PER_DEGREE,
        "follower_delta_deg": (actual - follower_zero) / TICKS_PER_DEGREE,
        # Actual was sampled BEFORE this cycle's command; compare with previous command.
        "tracking_error_deg": (previous - actual) / TICKS_PER_DEGREE,
        "travel_limited": abs(travel_goal - requested) > 1e-6,
        "boundary_limited": abs(bounded - travel_goal) > 1e-6,
        "session_limited": abs(session_goal - bounded) > 1e-6,
        "rate_limited": abs(goal - session_goal) > 1e-6,
    }


class Alignment:
    """Hold the last command during alignment; never recenter safety limits."""
    def __init__(self, names, master, follower, bounds, settings):
        self.names = list(names)
        self.settings = settings
        self.master_zero = master.copy()
        self.master_virtual = master.copy()
        self.master_last_raw = master.copy()
        self.follower_zero = follower.copy()
        self.previous = {n: follower[n] for n in names}
        travel = settings['travel_deg']
        self.envelope = (bounds.copy() if travel is None else
                         {n: (math.ceil(max(bounds[n][0], follower[n]-travel*TICKS_PER_DEGREE)),
                              math.floor(min(bounds[n][1], follower[n]+travel*TICKS_PER_DEGREE))) for n in names})
        self.paused = True
        self.segment = 0
        self.history = deque()

    def observe(self, now, master, actual):
        for n in self.names:
            # Unwrap each consecutive reading, so travel beyond 180 degrees and
            # crossing encoder zero do not reverse the follower's requested motion.
            step = (master[n] - self.master_last_raw[n] + 2048) % 4096 - 2048
            self.master_virtual[n] += step
            self.master_last_raw[n] = master[n]
        self.history.append((now, master.copy(), actual.copy()))
        # Keep enough time coverage even if the loop cadence varies slightly.
        while len(self.history) > 1 and now - self.history[1][0] >= .5:
            self.history.popleft()

    def pause(self):
        self.paused = True
        self.history.clear()

    def resume(self, master, actual):
        if not self.paused:
            return False, '当前已在跟随；先按P暂停再对齐。'
        if len(self.history) < 2 or self.history[-1][0]-self.history[0][0] < .4:
            return False, '请先托稳黑臂约半秒，再按R。'
        for n in self.names:
            for index in (1, 2):
                values = [s[index][n] for s in self.history]
                # Conservative across encoder wrap: do not accept ambiguous stability.
                if max(values)-min(values) > 2*TICKS_PER_DEGREE:
                    return False, f'{LABELS[n]}仍在移动，请稳住半秒再按R。'
            if abs(actual[n]-self.previous[n]) > 4*TICKS_PER_DEGREE:
                return False, f'白臂{LABELS[n]}还未稳定到保持目标，暂不重新对齐。'
        self.master_zero = (self.master_virtual if self.settings['travel_deg'] is None else master).copy()
        # Anchor to the held command, not a noisy/sagging measured position.
        self.follower_zero = self.previous.copy()
        self.segment += 1
        self.paused = False
        return True, '已重新设定参考，开始跟随；白臂目标连续不跳变。'

    def targets(self, master, actual, dt):
        goals = {}
        for n in self.names:
            if abs(self.previous[n]-actual[n]) > 8*TICKS_PER_DEGREE:
                raise RuntimeError(f'{LABELS[n]}偏离保持/跟随目标超过8°，停止')
            if self.paused:
                goals[n] = self.previous[n]
            else:
                position = self.master_virtual[n] if self.settings['travel_deg'] is None else master[n]
                delta = position-self.master_zero[n] if self.settings['travel_deg'] is None else (position-self.master_zero[n]+2048)%4096-2048
                if self.settings['leader_limit_deg'] is not None and abs(delta) > self.settings['leader_limit_deg']*TICKS_PER_DEGREE:
                    raise RuntimeError(f'主臂{LABELS[n]}偏离参考超过{self.settings["leader_limit_deg"]}°，停止')
                goals[n] = next_target(position, self.master_zero[n], self.follower_zero[n],
                                       self.previous[n], actual[n], *self.envelope[n], dt,
                                       **{k:self.settings[k] for k in ('gain','travel_deg','speed_deg_s')})
        return goals


class GripperControl:
    """Clutched, relative gripper mapping in normalized travel percentage."""
    def __init__(self, leader_cal, follower_cal, leader_raw, follower_raw):
        self.leader_start = leader_raw
        self.follower_start = follower_raw
        self.previous = follower_raw
        self.leader_span = leader_cal.range_max - leader_cal.range_min
        self.follower_low = follower_cal.range_min
        self.follower_high = follower_cal.range_max
        self.follower_span = self.follower_high - self.follower_low
        if self.leader_span <= 0 or self.follower_span <= 0:
            raise RuntimeError('夹爪校准范围无效')
        if not self.follower_low <= follower_raw <= self.follower_high:
            raise RuntimeError('白臂夹爪当前位置超出校准范围，无法启用')

    def target(self, leader_raw, follower_raw, dt):
        if abs(self.previous - follower_raw) > .12*self.follower_span:
            raise RuntimeError('夹爪实际位置偏离目标超过行程12%，停止')
        # A full master handgrip sweep maps to the full follower gripper travel.
        delta = (leader_raw - self.leader_start)*self.follower_span/self.leader_span
        desired = max(self.follower_low, min(self.follower_high, self.follower_start+delta))
        step = self.follower_span*.15*min(dt,.05)
        goal = max(self.previous-step, min(self.previous+step, desired))
        return round(max(self.follower_low, min(self.follower_high, goal)))

    def reanchor(self, leader_raw):
        self.leader_start = leader_raw
        self.follower_start = self.previous


def print_summary(stats):
    print("\n诊断摘要（误差与上一周期已发送目标比较；不能单独证明校准准确）：")
    for name, s in stats.items():
        count = s["samples"]
        if count:
            flags = " / ".join(f"{label}{100*s[key]/count:.0f}%" for key, label in
                                  [("travel_limited", "参考点限幅"), ("session_limited", "本次总范围限幅"), ("boundary_limited", "校准边界限幅"),
                                   ("rate_limited", "速度限幅")])
            print(f"{LABELS[name]}：最大跟踪误差{s['max_error']:.2f}{'%' if name == 'gripper' else '°'}；{flags}")


def main():
    parser = argparse.ArgumentParser(description="默认只读；--run 在本机确认后启动小范围相对跟随")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--watch", action="store_true", help="每秒显示只读边界检查，Q退出")
    parser.add_argument("--log-dir", type=Path, default=Path.home() / "so101-follow-logs")
    parser.add_argument("--profile", choices=PROFILES, default=None,
                        help="full: 校准范围内1倍/30°每秒；wide: 1倍/±60°；expanded: ±30°；gentle: 0.3倍/±15°")
    parser.add_argument('--record', action='store_true', help='同步保存ACT数据（仍需--run启动）')
    parser.add_argument('--task', default='Pick up the small block and place it in the metal box.')
    parser.add_argument('--dataset-dir', type=Path, help='新数据集目录，必须不存在；默认自动生成时间戳')
    parser.add_argument('--episode-seconds', type=int, default=45)
    parser.add_argument('--camera', default=os.getenv('SO101_CAMERA'))
    parser.add_argument('--leader-port', default=os.getenv('SO101_LEADER_PORT'))
    parser.add_argument('--follower-port', default=os.getenv('SO101_FOLLOWER_PORT'))
    args = parser.parse_args()
    if not args.leader_port or not args.follower_port:
        parser.error('请通过 --leader-port/--follower-port 或环境变量配置两个串口')
    if args.record and not args.camera:
        parser.error('录制时请通过 --camera 或 SO101_CAMERA 配置相机')
    if not 3 <= args.episode_seconds <= 120:
        parser.error('--episode-seconds须在3到120之间')
    args.profile = args.profile or ('full' if args.record else 'expanded')
    settings = PROFILES[args.profile]
    leader = SO101Leader(SO101LeaderConfig(
        port=args.leader_port, id="black_leader"))
    follower = SO101Follower(SO101FollowerConfig(
        port=args.follower_port, id="white_follower"))
    opened = []
    terminal = None
    armed = False
    gripper = None
    log = None
    log_path = None
    recorder = None
    stop_reason = "not_started"
    stats = {}
    try:
        for device in (leader, follower):
            device.bus.connect()
            opened.append(device.bus)
            if not device.calibration or not device.bus.is_calibrated:
                raise RuntimeError(f"{device.id} 校准文件缺失或与舵机不一致")
            modes = device.bus.sync_read("Operating_Mode", normalize=False, num_retry=3)
            if any(v != 0 for v in modes.values()):
                raise RuntimeError(f"{device.id} 不是位置模式：{modes}")
            torques = device.bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
            if any(v != 0 for v in torques.values()):
                raise RuntimeError(f"{device.id} 力矩未关闭，请先退出其它控制程序")
        print("串口、校准一致性、位置模式、关闭力矩：检查通过。", flush=True)
        names = [n for n in follower.bus.motors if n != "gripper"]
        positions = follower.bus.sync_read("Present_Position", normalize=False, num_retry=3)
        report(follower, positions, names)
        if not args.run and not args.watch:
            print("只读检查结束，没有启动力矩或发送位置目标。")
            return
        if not sys.stdin.isatty():
            raise RuntimeError("运动测试必须在本机交互终端启动")
        if args.record and args.run:
            from so101_record import Recorder
            recorder = Recorder(args, leader, follower)
        total_limit = 1800 if recorder else 180
        active_limit = 1200 if recorder else 60
        print(f"档位 {args.profile}：相对跟随×{settings['gain']}；" +
              ("不设起点附近幅度限制；" if settings['travel_deg'] is None else f"每关节偏离起点最多{settings['travel_deg']}°；") +
              f"目标速度上限{settings['speed_deg_s']}°/秒。")
        print("清空白臂周围，黑臂用手托住。停止后白臂会失去支撑，请安排软垫。")
        print("回车：白臂原位保持；调整黑臂后按R开始跟随（自动启用夹爪）；P暂停对齐；空格/Q/Ctrl+C停止。")
        print("跟随后直接开合黑臂手柄控制白臂夹爪；G可在当前位置重新设定夹爪参考。")
        print(f"只手动调整黑臂，不要掰通电保持中的白臂。累计跟随最多{active_limit}秒，总时长最多{total_limit}秒。")
        terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        commands = CommandInput(sys.stdin.fileno())
        print("当前为只读预检，每秒刷新。" + ("准备好按回车让白臂原位保持（尚不跟随）；未通过可调整后再按回车。" if args.run else "只读监视，不会启动。") + " Q退出。")
        last_report = 0
        while True:
            key = commands.read(recorder.poll() if recorder else ())
            now = time.monotonic()
            if now - last_report >= 1:
                positions = follower.bus.sync_read("Present_Position", normalize=False, num_retry=3)
                report(follower, positions, names)
                last_report = now
            if key:
                if recorder:
                    recorder.preflight(key)
                if key in ("q", " "):
                    return
                if key in ("\n", "\r") and args.run:
                    positions = follower.bus.sync_read("Present_Position", normalize=False, num_retry=3)
                    if report(follower, positions, names) is not None:
                        print("请保持白臂稳定，正在复核当前位置…", flush=True)
                        time.sleep(.25)
                        fresh = follower.bus.sync_read("Present_Position", normalize=False, num_retry=3)
                        if any(abs(fresh[n] - positions[n]) > 2*TICKS_PER_DEGREE for n in names):
                            print("白臂仍在移动，未启动力矩。托稳后再按回车。")
                            if recorder:
                                recorder.notify('白臂仍在移动，未启动力矩；托稳后再次按回车。', warning=True)
                        elif report(follower, fresh, names) is not None:
                            break
                    elif recorder:
                        recorder.notify('启动检查未通过，请查看终端对应关节提示后再次按回车。', warning=True)
            time.sleep(.05)
        lzero = leader.bus.sync_read("Present_Position", normalize=False, num_retry=3)
        fzero = follower.bus.sync_read("Present_Position", normalize=False, num_retry=3)
        bounds = report(follower, fzero, names)
        if bounds is None:
            raise RuntimeError("启动瞬间白臂越界，未启动力矩，请重新运行")
        queued = commands.read(recorder.poll() if recorder else ())
        if queued == 'q':
            return
        if queued:
            commands.pending.appendleft(queued)
        seeded = {n: max(bounds[n][0], min(bounds[n][1], fzero[n])) for n in names}
        previous = {n: float(seeded[n]) for n in names}
        alignment = Alignment(names, lzero, seeded, bounds, settings)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')
        log = log_path.open('x', encoding='utf-8')
        log.write(json.dumps({"type": "metadata", "utc": datetime.now(timezone.utc).isoformat(),
                              "profile": args.profile, **settings,
                              "leader_start": lzero, "follower_start": fzero, "follower_seed_goal": seeded, "bounds": bounds,
                              "session_envelope": alignment.envelope,
                              "schema_version": 2, "startup_state": "holding",
                              "leader_calibration": {k: asdict(v) for k, v in leader.calibration.items()},
                              "follower_calibration": {k: asdict(v) for k, v in follower.calibration.items()},
                              "gripper": "G activates from current positions; relative normalized travel, 15% per second",
                              "sampling": "sequential leader/follower reads before command, no camera synchronization"}) + '\n')
        log.flush()
        stats = {n: {"samples": 0, "max_error": 0., "travel_limited": 0,
                     "boundary_limited": 0, "session_limited": 0, "rate_limited": 0} for n in names}
        print(f"诊断日志自动保存到：{log_path}", flush=True)
        # Seed current-position goals BEFORE torque enable, avoiding stale targets.
        follower.bus.sync_write("Goal_Position", seeded, normalize=False)
        armed = True
        enable_verified_torque(follower.bus, names)
        started = last = time.monotonic()
        last_display = started - 1
        active_seconds = 0.
        stop_reason = "session_duration_limit"
        print("白臂正在原位保持。现在可逐个调整黑臂，让两臂连杆大致平行。稳住半秒后按R开始跟随。", flush=True)
        if recorder:
            recorder.set_control('holding', '白臂已原位保持；托稳黑臂半秒，按 R 开始跟随。')
        while time.monotonic() - started < total_limit:
            cycle_started = time.monotonic()
            key = commands.read(recorder.poll() if recorder else ())
            if key in ("q", " "):
                stop_reason = "user_stop"
                if recorder:
                    recorder.set_control('stopped', '收到停止请求，正在结束控制并完成磁盘写入；未确认片段保留备份。')
                break
            if recorder:
                key = recorder.handle(key, alignment, gripper)
            now = cycle_started
            dt = now - last
            if dt > 0.3:
                raise RuntimeError("控制循环延迟超过0.3秒，停止")
            last = now
            if not alignment.paused:
                active_seconds += dt
            if active_seconds >= active_limit:
                stop_reason = 'active_duration_limit'
                break
            if key == 'p':
                alignment.pause()
                print('已暂停：白臂保持最后目标。调整黑臂后稳住半秒，按R重新开始。', flush=True)
                if recorder:
                    recorder.set_control('holding', '白臂已暂停保持。当前段可 S 保存；下一段请 R 跟随后 C 录制。')
                log.write(json.dumps({'type':'event', 'event':'pause', 'elapsed_s':now-started,
                                      'segment':alignment.segment})+'\n')
            master = leader.bus.sync_read("Present_Position", normalize=False, num_retry=3)
            actual = follower.bus.sync_read("Present_Position", normalize=False, num_retry=3)
            record_sample = recorder.observation(actual) if recorder else None
            alignment.observe(now, master, actual)
            # Persist readings before guards, including the frame that triggers an abort.
            sample = {"type": "read", "utc": datetime.now(timezone.utc).isoformat(),
                      "monotonic_s": now, "elapsed_s": now - started,
                      "read_duration_s": time.monotonic() - now, "dt_s": dt,
                      "state": 'holding' if alignment.paused else 'following', "segment": alignment.segment,
                      "master": master, "actual": actual, "previous_target": previous.copy()}
            log.write(json.dumps(sample) + '\n')
            log.flush()
            goals = {}
            details = {}
            for name in names:
                c = follower.calibration[name]
                if not max(35, c.range_min)-START_TOLERANCE_TICKS <= actual[name] <= min(4060, c.range_max)+START_TOLERANCE_TICKS:
                    raise RuntimeError(f"{LABELS[name]} 实际读数{actual[name]}越过校准/编码器边界，停止")
            if key == 'r':
                accepted, message = alignment.resume(master, actual)
                print(message, flush=True)
                if recorder:
                    recorder.set_control('following' if accepted else 'holding', message)
                log.write(json.dumps({'type':'event', 'event':'resume' if accepted else 'resume_rejected',
                                      'message':message, 'elapsed_s':now-started, 'segment':alignment.segment,
                                      'leader_reference':alignment.master_zero,
                                      'follower_target_reference':alignment.follower_zero,
                                      'follower_actual_at_reference':actual})+'\n')
                if accepted and gripper is not None:
                    gripper.reanchor(master['gripper'])
                    log.write(json.dumps({'type':'event','event':'gripper_reanchor_on_resume',
                                          'elapsed_s':now-started,
                                          'leader_start':master['gripper'],
                                          'follower_target':gripper.previous})+'\n')
            auto_gripper = key == 'r' and accepted and gripper is None
            if key == 'g' or auto_gripper:
                if gripper is None:
                    grip = GripperControl(leader.calibration['gripper'], follower.calibration['gripper'],
                                          master['gripper'], actual['gripper'])
                    # The gripper has been torque-free. Seed its current goal first.
                    follower.bus.sync_write('Goal_Position', {'gripper': actual['gripper']}, normalize=False)
                    enable_verified_torque(follower.bus, 'gripper')
                    gripper = grip
                    previous['gripper'] = grip.previous
                    stats['gripper'] = {'samples':0, 'max_error':0., 'travel_limited':0,
                                        'boundary_limited':0, 'session_limited':0, 'rate_limited':0}
                    print('夹爪已自动启用：缓慢开合黑臂手柄，白臂夹爪将跟随。' if auto_gripper else
                          '夹爪已启用：缓慢开合黑臂手柄，白臂夹爪将跟随。', flush=True)
                    log.write(json.dumps({'type':'event','event':'gripper_enabled','elapsed_s':now-started,
                                          'automatic':auto_gripper,
                                          'leader_start':master['gripper'],'follower_start':actual['gripper']})+'\n')
                else:
                    gripper.reanchor(master['gripper'])
                    print('夹爪参考已更新；白臂夹爪保持当前位置。', flush=True)
                    log.write(json.dumps({'type':'event','event':'gripper_reanchor','elapsed_s':now-started,
                                          'leader_start':master['gripper'],'follower_target':gripper.previous})+'\n')
            if recorder:
                recorder.set_control('holding' if alignment.paused else 'following', gripper_ready=gripper is not None)
            goals = alignment.targets(master, actual, dt)
            if gripper is not None:
                if not gripper.follower_low <= actual['gripper'] <= gripper.follower_high:
                    raise RuntimeError('白臂夹爪实际位置越过校准范围，停止')
                goal = gripper.previous if alignment.paused else gripper.target(master['gripper'], actual['gripper'], dt)
                goals['gripper'] = goal
                details['gripper'] = {
                    'master_raw':master['gripper'], 'actual_raw':actual['gripper'],
                    'previous_target_raw':gripper.previous, 'command_raw':goal,
                    'master_delta_pct':100*(master['gripper']-gripper.leader_start)/gripper.leader_span,
                    'follower_delta_pct':100*(actual['gripper']-gripper.follower_start)/gripper.follower_span,
                    'tracking_error_pct':100*(gripper.previous-actual['gripper'])/gripper.follower_span,
                    'tracking_error_deg':0.,
                    'travel_limited':False, 'boundary_limited':False, 'session_limited':False,
                    'rate_limited':not alignment.paused and abs(goal-(gripper.follower_start+
                                          (master['gripper']-gripper.leader_start)*gripper.follower_span/gripper.leader_span))>1,
                }
            for name in names:
                if not alignment.paused:
                    position = alignment.master_virtual[name] if settings['travel_deg'] is None else master[name]
                    details[name] = diagnostic(position, alignment.master_zero[name], alignment.follower_zero[name], previous[name],
                                              actual[name], *bounds[name], goals[name], gain=settings['gain'], travel_deg=settings['travel_deg'],
                                              session_low=alignment.envelope[name][0], session_high=alignment.envelope[name][1])
                    details[name]['master_raw'] = master[name]
                    if settings['travel_deg'] is None:
                        details[name]['master_unwrapped'] = position
                else:
                    position = alignment.master_virtual[name] if settings['travel_deg'] is None else master[name]
                    delta = position-alignment.master_zero[name] if settings['travel_deg'] is None else (position-alignment.master_zero[name]+2048)%4096-2048
                    details[name] = {'master_raw':master[name], 'actual_raw':actual[name],
                                     'previous_target_raw':previous[name], 'command_raw':round(goals[name]),
                                     'master_delta_deg':delta/TICKS_PER_DEGREE,
                                     'follower_delta_deg':(actual[name]-alignment.follower_zero[name])/TICKS_PER_DEGREE,
                                     'tracking_error_deg':(previous[name]-actual[name])/TICKS_PER_DEGREE,
                                     'travel_limited':False, 'boundary_limited':False, 'session_limited':False, 'rate_limited':False}
            if time.monotonic() - now > 0.3:
                raise RuntimeError("通信延迟过大，停止")
            sent_goals = {n: round(v) for n, v in goals.items()}
            follower.bus.sync_write("Goal_Position", sent_goals, normalize=False)
            if recorder:
                recorder.sent(record_sample, sent_goals)
            log.write(json.dumps({"type": "command", "elapsed_s": now - started,
                                  "state": 'holding' if alignment.paused else 'following', "segment": alignment.segment,
                                  "utc": datetime.now(timezone.utc).isoformat(), "joints": details}) + '\n')
            log.flush()
            for name, d in details.items():
                if alignment.paused and name != 'gripper':
                    continue
                s = stats[name]
                s["samples"] += 1
                s["max_error"] = max(s["max_error"], abs(d["tracking_error_pct"] if name == 'gripper' else d["tracking_error_deg"]))
                for flag in ("travel_limited", "boundary_limited", "session_limited", "rate_limited"):
                    s[flag] += int(d[flag])
            if now - last_display >= 1:
                limited = [f"{LABELS[n]}:" + ','.join(label for flag, label in
                           [("travel_limited", f"{settings['travel_deg']}°幅度" if settings['travel_deg'] is not None else '幅度'), ("boundary_limited", "校准边界"), ("session_limited", "本次总范围"), ("rate_limited", "速度")]
                           if d[flag]) for n, d in details.items()
                           if any(d[f] for f in ("travel_limited", "boundary_limited", "session_limited", "rate_limited"))]
                max_error = max(abs(d["tracking_error_deg"]) for d in details.values())
                state_label = '保持中，调整黑臂后按R' if alignment.paused else '跟随中，P暂停对齐'
                grip_status = (' | 夹爪未启用' if gripper is None else
                               f" | 夹爪{'保持' if alignment.paused else '跟随'}：黑手柄Δ{100*(master['gripper']-gripper.leader_start)/gripper.leader_span:+.0f}% 白夹爪Δ{100*(actual['gripper']-gripper.follower_start)/gripper.follower_span:+.0f}%")
                print(f"{state_label} | 总计{now-started:.0f}/{total_limit}s 跟随{active_seconds:.0f}/{active_limit}s | 最大误差{max_error:.2f}° | 限幅：{'；'.join(limited) or '无'}{grip_status}", flush=True)
                if recorder:
                    print('[采集状态] '+recorder.status_line(), flush=True)
                last_display = now
            previous = {n: round(v) for n, v in goals.items()}
            alignment.previous = previous.copy()
            if gripper is not None:
                gripper.previous = previous['gripper']
            time.sleep(max(0, 0.05 - (time.monotonic() - now)))
    except BaseException as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Release torque before terminal/log cleanup, including on logging failures.
        for bus in reversed(opened):
            try:
                bus.disconnect(disable_torque=armed and bus is follower.bus)
            except Exception as exc:
                print(f"断开失败：{exc}。若白臂仍有力矩，请切断白臂外部电源。", file=sys.stderr)
        if armed:
            print("跟随测试已结束，已尝试关闭白臂力矩。")
        try:
            if terminal is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal)
            if log is not None:
                try:
                    log.write(json.dumps({"type": "end", "reason": stop_reason, "stats": stats}) + '\n')
                finally:
                    log.close()
                print(f"日志已保存：{log_path}")
                print_summary(stats)
        finally:
            if recorder is not None:
                recorder.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户停止。")
    except (RuntimeError, ConnectionError, OSError) as exc:
        print(f"未启动或已停止：{exc}", file=sys.stderr)
        sys.exit(1)
