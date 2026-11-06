#!/usr/bin/env python3
"""ACT rollout: offline check, read-only live preview, explicitly armed execution."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import termios
import threading
import time
import tty

import numpy as np

DATASET = Path.home()/'so101-datasets'/'act_pick_place_50'
V2_DATASET = Path.home()/'so101-datasets-v2'/'act_pick_place_v2_60'
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
PERIOD = .05
MAX_ACTION_AGE = .5


class Model:
    def __init__(self, checkpoint, device, threads=4, temporal_ensemble=False):
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        torch.set_num_threads(threads)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA不可用，请使用 --device cpu')
        self.torch, self.device = torch, device
        self.temporal_ensemble = temporal_ensemble
        cfg = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
        cfg.device = device
        cfg.pretrained_backbone_weights = None  # Full checkpoint already contains backbone.
        if temporal_ensemble:
            cfg.n_action_steps = 1
            cfg.temporal_ensemble_coeff = .01
        assert cfg.input_features['observation.images.front'].shape == (3, 480, 640)
        assert cfg.output_features['action'].shape == (6,)
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=cfg, strict=True).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=checkpoint,
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})

    def _batch(self, state, rgb):
        t = self.torch
        image = t.as_tensor(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.
        return {'observation.state': t.as_tensor(np.array(state, dtype=np.float32)),
                'observation.images.front': image}

    def predict(self, state, rgb):
        t = self.torch
        with t.inference_mode(), (t.autocast('cuda', dtype=t.float16)
                                  if self.device == 'cuda' else nullcontext()):
            chunk = self.policy.predict_action_chunk(self.pre(self._batch(state, rgb)))
            actions = self.post(chunk)[0].cpu().numpy()
        if actions.shape != (100, 6) or not np.isfinite(actions).all():
            raise RuntimeError('模型输出形状错误或含NaN/Inf')
        return actions

    def predict_one(self, state, rgb):
        if not self.temporal_ensemble:
            raise RuntimeError('当前模型未启用ACT时间集成')
        t = self.torch
        with t.inference_mode(), (t.autocast('cuda', dtype=t.float16)
                                  if self.device == 'cuda' else nullcontext()):
            action = self.policy.select_action(self.pre(self._batch(state, rgb)))
            result = self.post(action)[0].cpu().numpy()
        if result.shape != (6,) or not np.isfinite(result).all():
            raise RuntimeError('ACT时间集成输出无效')
        return result

    def reset(self):
        self.policy.reset()


class Worker:
    """Latest observation only; timestamped chunks never queue up behind the robot."""
    def __init__(self, model):
        self.model, self.observation, self.result, self.error = model, None, None, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        last = -1.
        try:
            while not self.stop.is_set():
                obs = self.observation
                if obs is None or obs[0] == last:
                    self.stop.wait(.005)
                    continue
                started = time.monotonic()
                last, state, rgb = obs
                chunk = self.model.predict(state, rgb)
                self.result = (last, chunk, time.monotonic()-started)
                self.stop.wait(max(0, .25-(time.monotonic()-started)))
        except BaseException as exc:
            self.error = repr(exc)

    def target(self, now, first_action=False):
        if self.error:
            raise RuntimeError('推理线程失败：'+self.error)
        if self.result is None:
            return None
        stamp, chunk, latency = self.result
        age = now-stamp
        if age > MAX_ACTION_AGE:
            return None
        # While stationary, future chunk steps have not been executed. Checking
        # an age-offset step here falsely treats planned motion as a start jump.
        index = 0 if first_action else min(int(max(0., age)/PERIOD), len(chunk)-1)
        return chunk[index], age, latency


class JointSmoother:
    """Low-pass only the five arm joints; preserve gripper timing and all guards."""
    def __init__(self, tau_s):
        if not 0 <= tau_s <= .5:
            raise ValueError('平滑时间常数须在0至0.5秒之间')
        self.tau_s = tau_s
        self.value = None

    def reset(self, actual):
        actual = np.asarray(actual, dtype=float)
        if actual.shape != (6,) or not np.isfinite(actual).all():
            raise RuntimeError('平滑器起点无效')
        self.value = actual.copy()

    def update(self, desired, dt):
        desired = np.asarray(desired, dtype=float)
        if desired.shape != (6,) or not np.isfinite(desired).all() or not 0 < dt <= .2:
            raise RuntimeError('平滑器输入无效')
        if self.value is None or self.tau_s == 0:
            self.value = desired.copy()
        else:
            alpha = -math.expm1(-min(dt, PERIOD)/self.tau_s)
            self.value[:5] += alpha * (desired[:5]-self.value[:5])
            self.value[5] = desired[5]
        return self.value.copy()


class TrialVideo:
    """Write the already-open camera stream off the motor-control thread."""
    def __init__(self, path):
        self.path = Path(path)
        self.frames = queue.Queue(maxsize=128)
        self.thread = None
        self.writer = None
        self.count = 0
        self.error = None

    def add(self, rgb):
        import cv2
        if self.error is not None:
            raise RuntimeError(f'试跑录像写入失败：{self.error}')
        if self.thread is None:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'),
                                          20, (640, 480))
            if not self.writer.isOpened():
                raise RuntimeError('无法创建试跑录像文件')
            self.thread = threading.Thread(target=self._write, daemon=False)
            self.thread.start()
        if rgb.shape != (480, 640, 3):
            raise RuntimeError('试跑录像画面尺寸不正确')
        try:
            self.frames.put_nowait(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except queue.Full:
            raise RuntimeError('试跑录像写入跟不上相机帧率') from None

    def _write(self):
        try:
            while True:
                frame = self.frames.get()
                if frame is None:
                    break
                self.writer.write(frame)
                self.count += 1
        except BaseException as exc:
            self.error = repr(exc)
        finally:
            self.writer.release()

    def close(self):
        if self.thread is None:
            return 0
        self.frames.put(None, timeout=5)
        self.thread.join(timeout=15)
        if self.thread.is_alive() or self.error is not None:
            raise RuntimeError(f'试跑录像未正常写完：{self.error or "超时"}')
        if self.count == 0 or not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError('试跑录像为空')
        return self.count


def bounded_goal(desired, actual, previous, low, high, dt):
    """Absolute actions in degrees / percent; preserve collection-time speed limits."""
    arrays = [np.asarray(a, dtype=float) for a in (desired, actual, previous, low, high)]
    if any(a.shape != (6,) or not np.isfinite(a).all() for a in arrays):
        raise RuntimeError('位置数据无效')
    desired, actual, previous, low, high = arrays
    if not 0 < dt <= .2 or np.any(low >= high):
        raise RuntimeError('控制周期或关节范围无效')
    if np.any(np.abs(actual-previous) > np.array([8]*5+[15])):
        raise RuntimeError('白臂未跟上上一目标，停止')
    if np.any(actual < low-1) or np.any(actual > high+1):
        raise RuntimeError('白臂当前位置超出允许范围')
    target = np.clip(desired, low, high)
    step = np.array([30]*5+[15]) * min(dt, PERIOD)
    goal = np.clip(target, previous-step, previous+step)
    return np.clip(goal, low, high)


def startup_check(desired, actual):
    if any(np.asarray(a).shape != (6,) or not np.isfinite(a).all() for a in (desired, actual)):
        raise RuntimeError('启动位置数据无效')
    delta = np.abs(np.asarray(desired)-np.asarray(actual))
    if np.any(delta > 15):
        details = [f'{NAMES[i]}差{delta[i]:.1f}{"%" if i==5 else "°"}' for i in np.where(delta > 15)[0]]
        raise RuntimeError('模型预测的第一步目标与当前读数相差超过15°/15%：'+', '.join(details)+
                           '；这与示范起姿检查是两项不同检查，请检查模型预测和相机画面')


def window(commands, frames):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    root = tk.Tk()
    root.title('SO101 ACT 推理预览')
    root.geometry('1200x950')
    status = tk.Label(root, bg='white')
    status.pack(fill='x')
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 23) if Path(font_path).exists() else ImageFont.load_default()
    canvas = tk.Label(root, bg='#18212b')
    canvas.pack(fill='both', expand=True)
    bar = tk.Frame(root)
    bar.pack(fill='x')
    for title, key in [('Start / Resume [R]', 'r'), ('Hold [P]', 'p'), ('STOP [Q / Space]', 'q')]:
        tk.Button(bar, text=title, font=('Sans', 15), command=lambda k=key: commands.put(k)).pack(side='left', expand=True, fill='x')
    root.bind('<KeyPress>', lambda e: commands.put(e.keysym.lower()))
    root.protocol('WM_DELETE_WINDOW', lambda: (commands.put('q'), root.destroy()))
    latest = None
    def tick():
        nonlocal latest
        try:
            while True:
                latest = frames.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            text, rgb = latest
            banner = Image.new('RGB', (max(800, root.winfo_width()), 90), 'white')
            ImageDraw.Draw(banner).multiline_text((12, 8), text, fill='black', font=font, spacing=5)
            status.image = ImageTk.PhotoImage(banner)
            status.configure(image=status.image)
            im = Image.fromarray(rgb)
            scale = min(max(640, canvas.winfo_width())/640, max(480, canvas.winfo_height())/480)
            im = im.resize((int(640*scale), int(480*scale)))
            canvas.image = ImageTk.PhotoImage(im)
            canvas.configure(image=canvas.image)
        root.after(100, tick)
    tick()
    root.mainloop()


def offline_check(args, model):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('local/act_pick_place_v2_60' if args.v2 else 'local/act_pick_place_50',
                        root=args.dataset, video_backend='pyav')
    times = []
    total = 60 if args.temporal_ensemble else 12
    for i in range(total):
        sample = ds[i if args.temporal_ensemble else (i*997) % len(ds)]
        rgb = (sample['observation.images.front'].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
        start = time.monotonic()
        result = (model.predict_one if args.temporal_ensemble else model.predict)(
            sample['observation.state'].numpy(), rgb)
        if i >= (10 if args.temporal_ensemble else 2):
            times.append(time.monotonic()-start)
        assert np.isfinite(result).all()
    report = {'checkpoint': args.checkpoint, 'device': args.device,
              'median_inference_s': float(np.median(times)),
              'p95_inference_s': float(np.percentile(times,95)), 'max_inference_s': max(times),
              'output_shape': list(result.shape), 'all_outputs_finite': True,
              'temporal_ensemble': args.temporal_ensemble,
              'timing_ok_for_control_cycle': max(times) < (.05 if args.temporal_ensemble else .2),
              'hardware_accessed': False}
    out = args.log_dir
    out.mkdir(exist_ok=True)
    (out/('offline_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def live(args, model):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from so101_record import Camera, normalized
    from so101_follow_test import report, enable_verified_torque
    from so101_record_ui import CommandInput, drain, replace_latest
    from so101_pose_check import load_starts, require_training_start
    import fcntl
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id='white_follower', use_degrees=True))
    expected = json.loads(Path(args.collection).read_text())['follower_calibration']
    if {k:asdict(v) for k,v in robot.calibration.items()} != expected:
        raise RuntimeError('当前校准与训练数据的校准不一致')
    if not sys.stdin.isatty():
        raise RuntimeError('请在交互终端运行')
    bus = robot.bus
    training_starts = load_starts(args.dataset)
    camera = worker = ui = terminal = log = video = None
    opened = armed = False
    lock = open('/tmp/so101-inference.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        bus.connect()
        opened = True
        # Exclusive tty ownership prevents a second serial opener during rollout.
        fcntl.ioctl(bus.port_handler.ser.fileno(), termios.TIOCEXCL)
        if not bus.is_calibrated:
            raise RuntimeError('舵机校准与文件不一致')
        if any(bus.sync_read('Torque_Enable', normalize=False, num_retry=2).values()):
            raise RuntimeError('白臂已有力矩，请先退出跟随/采集程序')
        if any(bus.sync_read('Operating_Mode', normalize=False, num_retry=2).values()):
            raise RuntimeError('舵机不是位置模式')
        raw = bus.sync_read('Present_Position', normalize=False, num_retry=2)
        bounds = report(robot, raw, NAMES)
        if bounds is None:
            raise RuntimeError('启动范围检查失败')
        low = normalized(bus, {n:bounds[n][0] for n in NAMES})
        high = normalized(bus, {n:bounds[n][1] for n in NAMES})
        low, high = np.minimum(low, high), np.maximum(low, high)
        camera = Camera(args.camera)
        camera.get(wait_s=5)
        if not args.temporal_ensemble:
            worker = Worker(model)
        ctx = mp.get_context('spawn')
        commands, frames = ctx.Queue(), ctx.Queue(maxsize=1)
        ui = ctx.Process(target=window, args=(commands, frames), daemon=True)
        ui.start()
        terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        keys = CommandInput(sys.stdin.fileno())
        folder = args.log_dir
        folder.mkdir(exist_ok=True)
        logpath = folder/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
        if args.record_video:
            video = TrialVideo(logpath.with_suffix('.mp4'))
        log = logpath.open('x')
        log.write(json.dumps({'type':'config', 'args':vars(args), 'calibration':expected,
                             'fps':20, 'replan_hz':20 if args.temporal_ensemble else 4,
                             'temporal_ensemble_coeff':.01 if args.temporal_ensemble else None,
                             'max_action_age':MAX_ACTION_AGE,
                             'video':str(video.path) if video else None})+'\n')
        print(f'日志：{logpath}\nR启动；P保持；Q/空格/Ctrl+C停止并卸力。退出后请承托白臂。', flush=True)
        if video:
            print(f'本次按R开始倒计时后录制第三视角：{video.path}', flush=True)
        previous = normalized(bus, raw)
        smoother = JointSmoother(args.smooth_ms/1000)
        paused, countdown, started = True, None, None
        last = time.monotonic()
        last_display = 0.
        while True:
            now = time.monotonic()
            dt = now-last
            last = now
            if armed and dt > .2:
                raise RuntimeError(f'控制循环延迟{dt:.3f}秒，停止')
            if not ui.is_alive():
                raise RuntimeError('预览窗口已关闭')
            key = keys.read(drain(commands))
            if key == 'q':
                break
            raw = bus.sync_read('Present_Position', normalize=False, num_retry=2)
            if armed and any(not bounds[n][0]-12 <= raw[n] <= bounds[n][1]+12 for n in NAMES):
                raise RuntimeError('编码器位置超出关节允许范围')
            actual = normalized(bus, raw)
            receipt, sequence, rgb = camera.get(max_age=.1)
            if args.temporal_ensemble:
                inference_start = time.monotonic()
                desired = (model.predict_one(actual, rgb) if armed and not paused else
                           model.predict(actual, rgb)[0])
                latency = time.monotonic()-inference_start
                target = (desired, latency, latency)
            else:
                worker.observation = (now, actual.copy(), rgb)
                target = worker.target(time.monotonic(), first_action=paused)
            if key == 'p':
                paused, countdown = True, None
                if armed:
                    previous = actual.copy()
                    bus.sync_write('Goal_Position', raw, normalize=False)
                    smoother.reset(actual)
                    if args.temporal_ensemble:
                        model.reset()
            if key == 'r':
                if not args.run:
                    print('当前仅预览。驱动白臂需要退出后使用 --run 启动。', flush=True)
                elif target is not None and paused:
                    if not armed:
                        require_training_start(actual, training_starts)
                    startup_check(target[0], actual)
                    countdown = time.monotonic()+3
            if countdown is not None and time.monotonic() >= countdown:
                if target is None:
                    raise RuntimeError('启动时模型没有及时给出动作')
                if not armed:
                    require_training_start(actual, training_starts)
                startup_check(target[0], actual)
                previous = actual.copy()
                smoother.reset(actual)
                if args.temporal_ensemble:
                    model.reset()
                bus.sync_write('Goal_Position', raw, normalize=False)
                armed = True  # Cleanup must run even if enabling torque partially fails.
                enable_verified_torque(bus, NAMES)
                paused, countdown = False, None
                started = started or time.monotonic()
            if started and time.monotonic()-started >= args.seconds:
                break
            if video and (countdown is not None or started is not None):
                video.add(rgb)
            filtered = None
            if armed and not paused:
                if target is None:
                    raise RuntimeError('模型动作过期或推理超时，停止')
                if time.monotonic()-now > .15:
                    raise RuntimeError('读取设备耗时过长，停止')
                filtered = smoother.update(target[0], max(.001, dt))
                goal = bounded_goal(filtered, actual, previous, low, high, max(.001, dt))
                bus.sync_write('Goal_Position', dict(zip(NAMES, goal.tolist())), normalize=True)
                previous = goal
            status = ('运行中' if armed and not paused else '保持中' if armed else '只读预览')
            if countdown is not None:
                status = f'{max(0,countdown-time.monotonic()):.1f}秒后启动'
            log.write(json.dumps({'t':now, 'state':actual.tolist(),
                'prediction':None if target is None else target[0].tolist(),
                'filtered_prediction':None if filtered is None else filtered.tolist(),
                'sent_goal':previous.tolist() if armed else None,
                'action_age_s':None if target is None else target[1],
                'inference_s':None if target is None else target[2], 'status':status})+'\n')
            if now-last_display > .2:
                message = status + (' | 模型等待/延迟' if target is None else f' | 推理{target[2]*1000:.0f}ms / 动作年龄{target[1]*1000:.0f}ms')
                message += '\nR启动，P保持，Q/空格停止卸力；关节30°/秒，夹爪15%/秒'
                replace_latest(frames, (message, rgb))
                log.flush()
                last_display = now
            time.sleep(max(0, PERIOD-(time.monotonic()-now)))
    except BaseException as exc:
        if log:
            try:
                log.write(json.dumps({'type':'error', 'reason':f'{type(exc).__name__}: {exc}'}, ensure_ascii=False)+'\n')
                log.flush()
            except Exception:
                pass
        raise
    finally:
        if opened:
            try:
                bus.disconnect(disable_torque=armed)
            except Exception as exc:
                print(f'断开失败：{exc}；请切断白臂外部电源。', file=sys.stderr)
        if terminal is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal)
        if worker:
            worker.stop.set()
            worker.thread.join(timeout=2)
        if camera:
            camera.close()
        if ui:
            ui.terminate()
            ui.join(timeout=2)
        if log:
            log.close()
        if video:
            try:
                frames_saved = video.close()
                if frames_saved:
                    print(f'本次第三视角录像已保存：{video.path}（{frames_saved}帧）', flush=True)
            except Exception as exc:
                print(f'试跑录像保存失败：{exc}', file=sys.stderr)
        lock.close()
        if armed:
            print('已尝试关闭白臂力矩，请承托机械臂。', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='offline checkpoint and timing check')
    mode.add_argument('--preview', action='store_true', help='read camera/joints without motor commands')
    mode.add_argument('--run', action='store_true', help='allow R to arm the follower')
    p.add_argument('--v2', action='store_true', help='使用新场景60集和新训练模型')
    p.add_argument('--checkpoint', required=True, help='训练输出的 pretrained_model 目录')
    p.add_argument('--dataset-root', type=Path, help='对应训练数据集目录')
    p.add_argument('--log-dir', type=Path, default=Path.home()/'so101-inference-logs')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--port', default=os.getenv('SO101_FOLLOWER_PORT'))
    p.add_argument('--camera', default=os.getenv('SO101_CAMERA'))
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--smooth-ms', type=float, default=0,
                   help='前五个关节的目标平滑时间常数；建议先试150，0为原行为；夹爪不变')
    p.add_argument('--record-video', action='store_true',
                   help='仅本次真机试跑保存第三视角MP4；从按R倒计时开始')
    p.add_argument('--temporal-ensemble', action='store_true',
                   help='GPU上每个20Hz控制周期预测并使用ACT时间集成（系数0.01）')
    args = p.parse_args()
    args.dataset = str((args.dataset_root or (V2_DATASET if args.v2 else DATASET)).expanduser().resolve())
    args.collection = str(Path(args.dataset)/'collection_config.json')
    args.log_dir = args.log_dir.expanduser()
    if not args.check and (not args.port or not args.camera):
        p.error('真机预览/运行需要 --port 和 --camera，或相应环境变量')
    if not Path(args.dataset, 'meta', 'info.json').is_file():
        p.error('找不到对应训练数据集，请指定 --dataset-root')
    if not 1 <= args.threads <= 8 or not 1 <= args.seconds <= 120:
        p.error('threads须1至8，seconds须1至120')
    if not 0 <= args.smooth_ms <= 500:
        p.error('smooth-ms须0至500')
    if args.record_video and not args.run:
        p.error('--record-video只用于--run')
    if args.temporal_ensemble and args.device != 'cuda':
        p.error('--temporal-ensemble需要--device cuda')
    # Resolve once: an ongoing training job may later update the last symlink.
    args.checkpoint = str(Path(args.checkpoint).resolve(strict=True))
    print('本次使用模型：'+args.checkpoint, flush=True)
    model = Model(args.checkpoint, args.device, args.threads,
                  temporal_ensemble=args.temporal_ensemble)
    if args.check:
        offline_check(args, model)
    else:
        live(args, model)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('用户停止。')
    except Exception as exc:
        print(f'未启动或已停止：{exc}', file=sys.stderr)
        raise SystemExit(1)
