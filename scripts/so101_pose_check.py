#!/usr/bin/env python3
"""Read-only comparison with actual recorded episode starts. Never sends motor writes."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT = Path.home()/'so101-datasets'/'act_pick_place_50'
V2_ROOT = Path.home()/'so101-datasets-v2'/'act_pick_place_v2_60'
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
LABELS = ['底座转向', '肩部俯仰（大臂）', '肘部弯曲', '手腕俯仰', '手腕旋转', '夹爪']
TOLERANCE = np.array([5., 5., 5., 5., 6., 5.])


def load_starts(dataset_root=ROOT):
    import pyarrow.parquet as pq
    result = []
    for p in sorted((Path(dataset_root)/'data').rglob('*.parquet')):
        data = pq.read_table(p, columns=['episode_index', 'frame_index', 'index', 'observation.state']).to_pydict()
        for e, f, i, state in zip(data['episode_index'], data['frame_index'], data['index'], data['observation.state']):
            if f == 0:
                result.append({'episode':int(e), 'index':int(i), 'state':state})
    result.sort(key=lambda x:x['episode'])
    expected_count = json.loads((Path(dataset_root)/'meta'/'info.json').read_text())['total_episodes']
    if len(result) != expected_count or {r['episode'] for r in result} != set(range(expected_count)):
        raise RuntimeError('示范起始帧不完整')
    return result


def compare(actual, target):
    actual, target = np.asarray(actual), np.asarray(target)
    if actual.shape != (6,) or target.shape != (6,) or not np.isfinite([actual, target]).all():
        raise RuntimeError('关节数据无效')
    delta = target-actual
    return delta, bool(np.all(abs(delta) <= TOLERANCE))


def nearest(actual, starts):
    # Match one real six-joint posture, never combine independent joint extrema.
    errors = abs(np.asarray([r['state'] for r in starts])-actual)/TOLERANCE
    return min(range(len(starts)), key=lambda i:(errors[i].max(), np.linalg.norm(errors[i])))


def require_training_start(actual, starts):
    ref = starts[nearest(actual, starts)]
    delta, ok = compare(actual, ref['state'])
    if not ok:
        details = '；'.join(f'{LABELS[i]}需{delta[i]:+.1f}{"%" if i==5 else "°"}' for i in range(6) if abs(delta[i]) > TOLERANCE[i])
        raise RuntimeError(f'当前姿态偏离示范起点（最近为第{ref["episode"]+1}段）：{details}。请先运行 so101_pose_check.py 对齐')
    return ref['episode']


def reference_image(ref, dataset):
    return (dataset[ref['index']]['observation.images.front'].permute(1,2,0).numpy()*255).round().astype(np.uint8)


def render(actual, ref, rgb, stable=False, notice=''):
    from PIL import Image, ImageDraw, ImageFont
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    small = ImageFont.truetype(font, 25)
    large = ImageFont.truetype(font, 31)
    im = Image.new('RGB', (1300, 1000), '#f2f5f8')
    d = ImageDraw.Draw(im)
    delta, ok = compare(actual, ref['state'])
    title = '当前关节姿态接近该示范起点' if stable else '误差已达标，等待稳定1秒' if ok else '请手动调整白臂，工具不会驱动电机'
    d.text((20,12), title, font=large, fill='#087c35' if ok else '#a33b15')
    d.text((20,62), f'固定参考：第 {ref["episode"]+1} 段真实起始帧（以下画面来自数据集）', font=small, fill='black')
    image = Image.fromarray(rgb).resize((640,480))
    im.paste(image,(20,110))
    d.text((690,110), '关节          当前 → 示范       需调整',font=small,fill='black')
    for i in range(6):
        unit = '%' if i == 5 else '°'
        color = '#087c35' if abs(delta[i])<=TOLERANCE[i] else '#b53220'
        d.text((690,170+i*68), LABELS[i], font=small, fill=color)
        d.text((690,200+i*68), f'{actual[i]:7.1f} → {ref["state"][i]:7.1f}    {delta[i]:+6.1f}{unit}', font=small, fill=color)
    lines = ['判定条件：同一段示范的手腕旋转在±6°内，其他四个关节在±5°内，夹爪在±5个百分点内。',
             '这是排查起点差异的容差，不保证抓取成功；物块、盒子和相机还需匹配。',
             '红色需调整；绿色已接近。正数=读数需要增大，负数=读数需要减小。',
             '用鼠标选择示范段；N重新选最近起点；S保存当前检查；Q退出。',
             '未启动力矩：请承托白臂，避免为了对齐而强推关节。',notice]
    for i,line in enumerate(lines):
        d.text((20,650+i*48),line,font=small,fill='#25364a')
    return im


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default=os.getenv('SO101_FOLLOWER_PORT'))
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path.home()/'so101-pose-checks')
    p.add_argument('--log-dir',type=Path,default=Path.home()/'so101-inference-logs')
    p.add_argument('--episode',type=int,help='参考示范编号1至50；默认选择最接近当前姿态的一段并锁定')
    p.add_argument('--offline',action='store_true',help='用最近一次试跑起点离线生成对比图；不连接设备')
    p.add_argument('--v2',action='store_true',help='使用新场景60集的起始姿态')
    args=p.parse_args()
    dataset_root = (args.dataset_root or (V2_ROOT if args.v2 else ROOT)).expanduser().resolve()
    if not (dataset_root/'meta'/'info.json').is_file():
        p.error('找不到数据集，请指定 --dataset-root')
    count = json.loads((dataset_root/'meta'/'info.json').read_text())['total_episodes']
    collection = dataset_root/'collection_config.json'
    if not args.offline and not args.port:
        p.error('实时姿态检查需要 --port 或 SO101_FOLLOWER_PORT')
    if args.episode is not None and not 1<=args.episode<=count:p.error(f'episode须1至{count}')
    starts=load_starts(dataset_root)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset('local/act_pick_place_v2_60' if args.v2 else 'local/act_pick_place_50',
                      root=dataset_root,video_backend='pyav')
    folder=args.output_dir.expanduser();folder.mkdir(parents=True,exist_ok=True)
    if args.offline:
        logs=sorted(args.log_dir.expanduser().glob('*.jsonl'),reverse=True)
        sample=None
        for log in logs:
            for line in log.open():
                row=json.loads(line)
                if row.get('status')=='运行中':sample=row['state'];break
            if sample is not None:break
        if sample is None:raise RuntimeError('没有历史试跑记录')
        ref=starts[args.episode-1 if args.episode else nearest(sample,starts)]
        dest=folder/'offline_comparison.png'
        render(sample,ref,reference_image(ref,ds),notice='离线示例：取自最近试跑的起点，不是当前实时姿态。').save(dest)
        print('历史试跑起点对比图：',dest)
        return
    import fcntl,termios
    import tkinter as tk
    from PIL import ImageTk
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from so101_record import normalized
    robot=SO101Follower(SO101FollowerConfig(port=args.port,id='white_follower',use_degrees=True))
    expected=json.loads(collection.read_text())['follower_calibration']
    if {k:asdict(v) for k,v in robot.calibration.items()} != expected:raise RuntimeError('校准与采集时不一致')
    bus=robot.bus
    try:
        bus.connect()
        fcntl.ioctl(bus.port_handler.ser.fileno(),termios.TIOCEXCL)
        if not bus.is_calibrated:raise RuntimeError('舵机与本地校准不一致')
        if any(bus.sync_read('Torque_Enable',normalize=False,num_retry=2).values()):
            raise RuntimeError('白臂已启动力矩，请先退出推理或跟随程序，再承托白臂进行检查')
        raw=bus.sync_read('Present_Position',normalize=False,num_retry=2)
        actual=normalized(bus,raw)
        chosen=args.episode-1 if args.episode else nearest(actual,starts)
        root=tk.Tk();root.title('SO101 起始姿态对齐（只读）');root.geometry('1300x1050')
        selector=tk.StringVar(value=str(chosen+1))
        bar=tk.Frame(root);bar.pack()
        tk.Label(bar,text=f'Reference episode (1-{count}):',font=('Sans',14)).pack(side='left')
        tk.Spinbox(bar,from_=1,to=count,textvariable=selector,width=6,font=('Sans',18)).pack(side='left')
        tk.Button(bar,text='Nearest [N]',command=lambda:selector.set(str(nearest(actual,starts)+1))).pack(side='left')
        display=tk.Label(root);display.pack(fill='both',expand=True)
        ref=None;rgb=None;since=None;notice='';last_snapshot=None
        def save(event=None):
            nonlocal notice
            if last_snapshot is None:return
            path=folder/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            path.write_text(json.dumps(last_snapshot,ensure_ascii=False,indent=2))
            notice='已保存：'+path.name
            print(path,flush=True)
        tk.Button(bar,text='Save check [S]',command=save).pack(side='left')
        tk.Button(bar,text='Exit [Q]',command=root.destroy).pack(side='left')
        root.bind('q',lambda e:root.destroy());root.bind('s',save)
        root.bind('n',lambda e:selector.set(str(nearest(actual,starts)+1)))
        def tick():
            nonlocal actual,ref,rgb,since,last_snapshot
            try:
                raw=bus.sync_read('Present_Position',normalize=False,num_retry=2)
                actual=normalized(bus,raw)
                try:index=int(selector.get())-1
                except ValueError:index=chosen
                index=max(0,min(count-1,index))
                if ref is None or ref['episode']!=index:
                    ref=starts[index];rgb=reference_image(ref,ds);since=None
                delta,ok=compare(actual,ref['state'])
                # Gripper normalization clips raw values: reject out-of-calibration reads explicitly.
                valid=all(robot.calibration[n].range_min-8<=raw[n]<=robot.calibration[n].range_max+8 for n in NAMES)
                if not valid:raise RuntimeError('编码器读数超出校准范围，请停止调整并检查')
                since=(since or time.monotonic()) if ok else None
                stable=since is not None and time.monotonic()-since>=1
                last_snapshot={'time':datetime.now().isoformat(),'episode_1based':index+1,
                    'actual':actual.tolist(),'target':ref['state'],'delta':delta.tolist(),
                    'within_tolerance':ok,'stable_1s':stable,'tolerance':TOLERANCE.tolist()}
                im=render(actual,ref,rgb,stable,notice)
                scale=min(max(650,display.winfo_width())/1300,max(500,display.winfo_height())/1000)
                im=im.resize((int(1300*scale),int(1000*scale)))
                display.image=ImageTk.PhotoImage(im);display.configure(image=display.image)
                root.after(100,tick)
            except Exception as exc:
                print('检查停止：',exc,flush=True);root.destroy()
        root.after(100,tick)
        root.mainloop()
    finally:
        if bus.is_connected:bus.disconnect(disable_torque=False)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
    except Exception as exc:
        print('未启动或已停止：',exc)
        raise SystemExit(1)
