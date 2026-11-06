#!/usr/bin/env python3
"""SO101 relative-teleoperation recorder. Run with the LeRobot virtualenv.

Uses the existing controller in so101_follow_test; never enables torque itself.
Software synchronization uses camera receipt time, not exposure/hardware triggers.
"""
import json
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from so101_record_ui import PreviewProcess, drain

FPS = 20
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def features():
    return {
        'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': [n+'.pos' for n in NAMES]},
        'action': {'dtype': 'float32', 'shape': (6,), 'names': [n+'.pos' for n in NAMES]},
        'observation.images.front': {'dtype': 'video', 'shape': (480, 640, 3),
                                     'names': ['height', 'width', 'channels']},
    }


def normalized(bus, raw):
    # Same coordinate convention as the installed follower's get_observation/send_action.
    ids = {bus.motors[n].id: raw[n] for n in NAMES}
    result = bus._normalize(ids)
    return np.asarray([result[bus.motors[n].id] for n in NAMES], dtype=np.float32)


def quality(samples):
    intervals = np.diff([s['state_time'] for s in samples])
    hz = float(1 / intervals.mean()) if len(intervals) and intervals.mean() > 0 else 0.
    max_gap = float(intervals.max()) if len(intervals) else 0.
    max_age = max((s['camera_age_s'] for s in samples), default=0.)
    reasons = []
    if len(samples) < 2*FPS:
        reasons.append('不足2秒')
    if len(intervals) and np.any(intervals <= 0):
        reasons.append('采样时间未严格递增')
    if not .95*FPS <= hz <= 1.05*FPS:
        reasons.append(f'实测采样率{hz:.2f}Hz偏离20Hz超过5%')
    if max_gap > .1:
        reasons.append(f'采样间断{max_gap:.3f}秒超过0.1秒')
    if max_age > .1:
        reasons.append('相机画面时间差超过0.1秒')
    return {'frames': len(samples), 'effective_hz': hz, 'max_gap_s': max_gap,
            'max_camera_age_s': max_age, 'accepted': not reasons, 'reasons': reasons}


class DatasetSink:
    """Bounded queue; disk writes/encoding never run on the motor-control thread."""
    def __init__(self, root, config):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.configs.video import RGBEncoderConfig
        self.root = Path(root)
        self.dataset = LeRobotDataset.create(
            repo_id='local/'+self.root.name, root=self.root, fps=FPS,
            robot_type='so101_follower', features=features(), use_videos=True,
            image_writer_threads=2, encoder_threads=2,
            rgb_encoder=RGBEncoderConfig(vcodec='h264', preset='fast', crf=20),
        )
        (self.root/'collection_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
        self.queue = queue.Queue(maxsize=64)
        self.error = None
        self.busy = False
        self.saved = 0
        self.archived = 0
        self.events = queue.Queue()
        self.raw_dir = None
        self.raw_log = None
        self.raw_count = 0
        self.raw_index = 0
        self.needs_resume = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def check(self):
        if self.error:
            raise RuntimeError(f'数据写入失败：{self.error}')

    def put(self, kind, value=None):
        self.check()
        try:
            self.queue.put_nowait((kind, value))
        except queue.Full:
            raise RuntimeError('数据写入积压，已停止；已写入的原始帧保留在 raw 目录') from None

    def _backup_frame(self, frame, timing):
        if self.raw_dir is None:
            self.raw_dir = self.root/'raw'/f'episode_{self.raw_index:06d}'
            self.raw_dir.mkdir(parents=True, exist_ok=False)
            self.raw_index += 1
            self.raw_count = 0
            self.raw_log = (self.raw_dir/'frames.jsonl').open('x', encoding='utf-8')
        filename = f'frame_{self.raw_count:06d}.jpg'
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame['observation.images.front'], cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError('原始画面备份编码失败')
        (self.raw_dir/filename).write_bytes(encoded.tobytes())
        self.raw_log.write(json.dumps({'image': filename, 'task': frame['task'],
                                      'state': frame['observation.state'].tolist(),
                                      'action': frame['action'].tolist(), 'timing': timing})+'\n')
        self.raw_log.flush()
        self.raw_count += 1

    def _finish_backup(self, status, detail=None):
        if self.raw_dir is None:
            return None
        self.raw_log.flush()
        os.fsync(self.raw_log.fileno())
        self.raw_log.close()
        path = self.raw_dir
        manifest = {'status': status, 'frames': self.raw_count, 'detail': detail,
                    'format': 'JPEG RGB images + frames.jsonl state/action/timing; not a training dataset'}
        (path/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        self.raw_dir = self.raw_log = None
        self.raw_count = 0
        return str(path)

    def _run(self):
        try:
            while True:
                kind, value = self.queue.get()
                if kind == 'frame':
                    frame = dict(value)
                    timing = frame.pop('_timing', None)
                    self._backup_frame(frame, timing)
                    if self.needs_resume:
                        from lerobot.datasets.lerobot_dataset import LeRobotDataset
                        from lerobot.configs.video import RGBEncoderConfig
                        self.dataset = LeRobotDataset.resume(
                            repo_id='local/'+self.root.name, root=self.root, video_backend='pyav',
                            image_writer_threads=2, encoder_threads=2,
                            rgb_encoder=RGBEncoderConfig(vcodec='h264', preset='fast', crf=20))
                        self.needs_resume = False
                    self.dataset.add_frame(frame)
                elif kind == 'save':
                    self.dataset.save_episode()
                    # Close parquet footers before announcing success. The next
                    # episode lazily resumes the writer in this worker thread.
                    self.dataset.finalize()
                    self.needs_resume = True
                    with (self.root/'quality.jsonl').open('a') as f:
                        f.write(json.dumps({'episode_index': self.saved, **value})+'\n')
                        f.flush()
                        os.fsync(f.fileno())
                    raw_path = self._finish_backup('saved_trainable', value)
                    self.saved += 1
                    self.events.put({'kind': 'saved', 'message': f'已保存第 {self.saved} 段训练数据；'
                                     f'{value["summary"]["effective_hz"]:.2f} Hz。文件已写入磁盘。', 'path': raw_path})
                    self.busy = False
                elif kind == 'archive':
                    path = self._finish_backup('needs_review', value)
                    self.dataset.clear_episode_buffer()
                    self.archived += 1
                    reasons = '；'.join(value.get('summary', {}).get('reasons', []))
                    self.events.put({'kind': 'archived', 'message': f'已保存待检查备份：{path}；未计入训练数据。原因：{reasons}',
                                     'path': path})
                    self.busy = False
                elif kind == 'discard':
                    path = self._finish_backup('discarded_by_user', value)
                    self.dataset.clear_episode_buffer()
                    self.events.put({'kind': 'discarded', 'message': '当前段已从训练数据中丢弃，可 R 跟随后重新录制。',
                                     'path': path})
                    self.busy = False
                elif kind == 'stop':
                    path = self._finish_backup('interrupted_unsaved')
                    if path:
                        print(f'未确认片段已保留原始备份：{path}（不计训练数据）', flush=True)
                    if self.dataset.has_pending_frames():
                        self.dataset.clear_episode_buffer()
                    break
        except BaseException as exc:
            self.error = exc
        finally:
            if self.raw_dir is not None:
                try:
                    self._finish_backup('writer_error', str(self.error))
                except BaseException:
                    if self.raw_log and not self.raw_log.closed:
                        self.raw_log.close()
            try:
                self.dataset.finalize()
            except BaseException as exc:
                self.error = self.error or exc

    def close(self):
        while self.worker.is_alive():
            try:
                self.queue.put(('stop', None), timeout=.5)
                break
            except queue.Full:
                pass
        last_message = 0.
        while self.worker.is_alive():
            self.worker.join(.5)
            if time.monotonic()-last_message > 10:
                print('机械臂控制已结束，正在完成数据写入…', flush=True)
                last_message = time.monotonic()
        self.check()


class Camera:
    def __init__(self, device):
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.device = device
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError('无法打开第三视角相机，请关闭原来的相机预览窗口')
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # This MJPG camera advertises 60 fps at 640x480; the dataset samples
            # every third frame at 20 Hz. Request its supported mode explicitly.
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            seq = 0
            while not self.stop.is_set():
                ok, bgr = cap.read()
                receipt = time.monotonic()
                if not ok:
                    raise RuntimeError('相机读取失败')
                if bgr.shape != (480, 640, 3):
                    raise RuntimeError(f'相机没有提供640×480画面：{bgr.shape}')
                seq += 1
                self.latest = (receipt, seq, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except BaseException as exc:
            self.error = str(exc)
        finally:
            cap.release()

    def get(self, min_sequence=None, max_age=.15, wait_s=0.):
        deadline = time.monotonic()+wait_s
        while True:
            if self.error:
                raise RuntimeError(self.error)
            frame = self.latest
            age = time.monotonic()-frame[0] if frame is not None else float('inf')
            if frame is not None and age <= max_age and (min_sequence is None or frame[1] != min_sequence):
                return frame
            if time.monotonic() >= deadline:
                seq = frame[1] if frame is not None else None
                raise RuntimeError(f'相机没有新画面：上一帧序号{min_sequence}，当前{seq}，画面距今{age:.3f}秒')
            time.sleep(.002)

    def close(self):
        self.stop.set()
        self.worker.join(timeout=1)


class Recorder:
    def __init__(self, args, leader, follower, *, camera_factory=Camera,
                 preview_factory=PreviewProcess, sink_factory=DatasetSink):
        self.ui = None
        self.camera = None
        self.sink = None
        self.event_file = None
        self.phase = 'idle'
        self.control = 'startup'
        self.gripper_ready = False
        self.alignment = None
        self.samples = []
        self.last_seq = None
        self.last_preview = 0.
        self.notice = '按回车或点击“原位保持”，然后 R 跟随、C 录制、S 保存。'
        self.warning = False
        self.last_result = ''
        self.fault_reason = None
        self.task = args.task
        self.max_seconds = args.episode_seconds
        self.bus = follower.bus
        self.path = args.dataset_dir or Path.home()/'so101-datasets'/datetime.now().strftime('pick_place_%Y%m%d_%H%M%S_%f')
        try:
            if shutil.disk_usage(Path.home()).free < 5*1024**3:
                raise RuntimeError('可用磁盘空间不足5GB，请先清理空间')
            self.camera = camera_factory(args.camera)
            deadline = time.monotonic()+5
            while self.camera.latest is None and not self.camera.error and time.monotonic() < deadline:
                time.sleep(.05)
            self.camera.get()
            self.ui = preview_factory()
            config = {'task': args.task, 'profile': args.profile, 'fps': FPS,
                      'camera': args.camera, 'width':640, 'height':480,
                      'robot_type':'so101_follower', 'robot_id':'white_follower', 'use_degrees':True,
                      'action_semantics':'follower absolute Goal_Position, after relative mapping/rate/position limits, normalized with installed bus',
                      'sync':'latest camera receipt before motor command; host monotonic times in quality.jsonl; not hardware exposure synchronization',
                      'leader_calibration':{k:asdict(v) for k,v in leader.calibration.items()},
                      'follower_calibration':{k:asdict(v) for k,v in follower.calibration.items()}}
            self.sink = sink_factory(self.path, config)
            self.event_file = (self.path/'recording_events.jsonl').open('x', encoding='utf-8')
            self.notify(self.notice)
            print(f'\nACT 数据保存到：{self.path}\nC开始一段；S成功保存；X丢弃；P暂停；Q退出。', flush=True)
            print('每段最多%d秒；保存时白臂保持。未通过检查或退出时未确认的片段保留原始备份。' % self.max_seconds)
        except BaseException:
            self.close()
            raise

    def poll(self):
        self.sink.check()
        for event in drain(self.sink.events):
            self.phase = 'idle'
            self.samples = []
            self.last_result = event['message']
            self.notify(event['message'], warning=event['kind'] == 'archived', event=event['kind'])
        if time.monotonic()-self.last_preview >= .1:
            frame = self.camera.latest
            self.ui.publish(self.snapshot(), None if frame is None else frame[2])
            self.last_preview = time.monotonic()
        return self.ui.keys()

    def notify(self, message, warning=False, event='notice'):
        self.notice, self.warning = message, warning
        self.last_preview = 0.
        print(f'[采集] {message}', flush=True)
        if self.event_file:
            self.event_file.write(json.dumps({'time': datetime.now().isoformat(), 'event': event,
                                              'phase': self.phase, 'control': self.control,
                                              'frames': len(self.samples), 'message': message},
                                             ensure_ascii=False)+'\n')
            self.event_file.flush()

    def status_line(self):
        controls = {'startup':'待原位保持', 'holding':'白臂保持', 'following':'白臂跟随', 'stopped':'控制结束'}
        phases = {'idle':'未录制', 'recording':'● 正在录制', 'review':'待保存', 'saving':'正在写入，请稍候'}
        return (f'{controls[self.control]} | {phases[self.phase]} | 当前{len(self.samples)}帧 '
                f'({len(self.samples)/FPS:.1f}秒) | 训练数据{self.sink.saved}段 / 待检查备份{self.sink.archived}段')

    def snapshot(self):
        allowed = ['q', 's']  # S always explains what happened, even before C.
        if self.control == 'startup':
            allowed += ['\n']
        elif self.control != 'stopped' and self.phase != 'saving':
            allowed += ['p']
            if self.phase == 'idle':
                allowed += ['r']
                if self.control == 'following' and self.gripper_ready:
                    allowed += ['c']
            else:
                allowed += ['x']
        camera_age = time.monotonic()-self.camera.latest[0] if self.camera.latest else float('inf')
        camera_note = ' | 相机画面暂停' if self.camera.error or camera_age > .5 else ''
        return {'summary': self.status_line()+camera_note, 'notice': self.notice,
                'warning': self.warning, 'path': f'保存目录：{self.path}', 'allowed': allowed,
                'phase': self.phase, 'control': self.control, 'frames': len(self.samples),
                'saved': self.sink.saved, 'archived': self.sink.archived}

    def set_control(self, state, message=None, gripper_ready=None):
        changed = self.control != state or (gripper_ready is not None and self.gripper_ready != gripper_ready)
        self.control = state
        if gripper_ready is not None:
            self.gripper_ready = gripper_ready
        if changed:
            self.last_preview = 0.
        if message:
            self.notify(message, event='control')

    def preflight(self, key):
        if key and key not in ('\n', '\r', 'q', ' '):
            self.notify(f'收到 {key.upper()}：尚未启用白臂保持，请先按回车或点击“原位保持”。', warning=True)

    def handle(self, key, alignment, gripper):
        key = key.lower()
        self.alignment = alignment
        self.control = 'holding' if alignment.paused else 'following'
        self.gripper_ready = gripper is not None
        if self.phase == 'recording' and time.monotonic()-self.started >= self.max_seconds:
            self.phase = 'review'
            alignment.pause()
            self.control = 'holding'
            self.notify('达到单段时长，已暂停；按 S 保存或 X 丢弃。')
        if key and self.phase == 'saving':
            self.notify(f'收到 {key.upper()}：正在写入磁盘，请等待“已保存”提示；Q 可结束并等待写入完成。')
            return ''
        if key == 'c':
            if self.phase != 'idle' or alignment.paused or gripper is None:
                self.notify('收到 C：请先保存当前段（如有），按 R 进入跟随后再按 C。', warning=True)
            elif shutil.disk_usage(self.path).free < 5*1024**3:
                raise RuntimeError('可用磁盘不足5GB，请结束采集')
            else:
                try:
                    self.camera.get(max_age=.1)
                except RuntimeError as exc:
                    self.notify(f'收到 C，但相机尚未就绪：{exc}', warning=True)
                    return ''
                self.samples = []
                self.last_seq = None
                self.fault_reason = None
                self.started = time.monotonic()
                self.phase = 'recording'
                self.notify('收到 C：● 开始录制；完成动作后按 S 保存。', event='record_started')
            return ''
        if key in ('s','x'):
            if self.phase not in ('recording','review'):
                if key == 's' and self.last_result:
                    self.notify(self.last_result+' 当前没有新的待保存片段。')
                else:
                    self.notify(f'收到 {key.upper()}：当前没有待保存片段。请确认按 C 后显示“正在录制”。', warning=True)
                return ''
            self.phase = 'review'
            alignment.pause()
            self.control = 'holding'
            if not self.samples:
                self.phase = 'idle'
                self.notify('当前段为 0 帧，未生成空数据集片段。请 R 跟随后重新 C 录制。', warning=True)
                return ''
            result = quality(self.samples)
            if self.fault_reason:
                result['accepted'] = False
                result['reasons'].append(self.fault_reason)
            kind = 'discard' if key == 'x' else ('save' if result['accepted'] else 'archive')
            self.sink.busy = True
            self.phase = 'saving'
            self.sink.put(kind, {'summary':result, 'samples':list(self.samples)})
            if kind == 'archive':
                self.notify('收到 S：正在保存待检查备份；'+ '；'.join(result['reasons']), warning=True, event='archive_requested')
            else:
                self.notify(f'收到 {key.upper()}：'+ ('正在写入训练数据，请等待保存完成。' if kind == 'save' else '正在丢弃当前段。'),
                            event=kind+'_requested')
            return ''
        if key in ('r','g') and self.phase in ('recording','review'):
            self.notify('请先 S 保存或 X 丢弃当前段，再重新对齐。', warning=True)
            return ''
        if key == 'p' and self.phase == 'recording':
            self.phase = 'review'
            self.notify('录制已暂停；按 S 保存或 X 丢弃。')
        return key

    def observation(self, actual):
        if self.phase != 'recording':
            return None
        state_time = time.monotonic()
        # A camera frame can arrive just after the motor read. Allow up to one
        # 60 fps camera period plus scheduling slack before declaring a dropout.
        try:
            receipt, seq, image = self.camera.get(min_sequence=self.last_seq, max_age=.1, wait_s=.025)
        except RuntimeError as exc:
            self.phase = 'review'
            self.fault_reason = str(exc)
            if self.alignment:
                self.alignment.pause()
                self.control = 'holding'
            self.notify(f'相机采集暂停：{exc}。已录片段仍保留；按 S 保存待检查备份。', warning=True, event='camera_fault')
            return None
        age = abs(state_time-receipt)
        if age > .1:
            raise RuntimeError(f'相机与关节采样相差{age:.3f}秒，当前段未保存')
        self.last_seq = seq
        return {'observation.state':normalized(self.bus, actual), 'observation.images.front':image,
                'task':self.task}, {'state_time':state_time, 'camera_receipt_time':receipt,
                                   'camera_age_s':age, 'camera_sequence':seq}

    def sent(self, sample, goals):
        if sample is None:
            return
        frame, timing = sample
        frame['action'] = normalized(self.bus, goals)
        timing['command_sent_time'] = time.monotonic()
        self.sink.put('frame', {**frame, '_timing': timing})
        self.samples.append(timing)

    def close(self):
        if self.camera:
            self.camera.close()
        try:
            if self.sink:
                self.sink.close()
                print(f'保存完成：训练数据{self.sink.saved}段，待检查备份{self.sink.archived}段，目录 {self.path}', flush=True)
        finally:
            if self.ui:
                self.ui.close()
                self.ui = None
            if self.event_file:
                self.event_file.close()
                self.event_file = None


if __name__ == '__main__':
    # Keep a single implementation of all motor guards and torque cleanup.
    from so101_follow_test import main
    sys.argv.insert(1, '--record')
    try:
        main()
    except KeyboardInterrupt:
        print('用户结束采集。')
    except (RuntimeError, ConnectionError, OSError, ValueError) as exc:
        print(f'未启动或采集已停止：{exc}', file=sys.stderr)
        sys.exit(1)
