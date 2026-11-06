#!/usr/bin/env python3
"""Second-scene collection launcher; each run gets a new isolated dataset."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',help='save the fixed scene reference only; no hardware access')
    parser.add_argument('--root',type=Path,default=Path.home()/'so101-datasets-v2')
    parser.add_argument('--preview-image',type=Path,help='当前场景预览图片；仅 --prepare 使用')
    parser.add_argument('--camera',default=os.getenv('SO101_CAMERA'))
    parser.add_argument('--leader-port',default=os.getenv('SO101_LEADER_PORT'))
    parser.add_argument('--follower-port',default=os.getenv('SO101_FOLLOWER_PORT'))
    args=parser.parse_args()
    root=args.root.expanduser()
    scene=root/'scene_reference.jpg'
    if args.prepare:
        if scene.exists():
            print('已有场景参考，保留：',scene)
            return
        if args.preview_image is None:
            parser.error('--prepare 需要 --preview-image')
        source=args.preview_image.expanduser()
        if not source.exists() or time.time()-source.stat().st_mtime>15:
            raise RuntimeError('请先打开相机预览并确定新场景，再保存参考')
        root.mkdir(exist_ok=True)
        shutil.copy2(source,scene)
        (root/'scene_manifest.json').write_text(json.dumps({
            'created_at':datetime.now().isoformat(), 'reference_image':str(scene),
            'camera':args.camera,
            'width':640,'height':480,'fps':20,'scene':'v2_fixed_table',
            'task':'Pick up the small block and place it in the metal box.',
            'note':'Use only this scene for round two. Do not combine with round-one data.'
        },ensure_ascii=False,indent=2))
        print('第二轮场景参考：',scene)
        return
    if not scene.exists():
        raise RuntimeError('尚未保存第二轮场景参考，请先运行 --prepare')
    if not args.camera or not args.leader_port or not args.follower_port:
        parser.error('录制需要 --camera、--leader-port、--follower-port，或相应环境变量')
    if not sys.stdin.isatty():
        raise RuntimeError('请在本机交互终端启动采集')
    output=root/datetime.now().strftime('pick_place_%Y%m%d_%H%M%S_%f')
    print('本次新数据目录：',output,flush=True)
    print('回车原位保持 → R跟随（自动启用夹爪）→ C录制 → S成功保存 / X丢弃；Q结束卸力。',flush=True)
    try:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('so101_record.py')),
                               '--run','--profile','full','--camera',args.camera,
                               '--leader-port',args.leader_port,'--follower-port',args.follower_port,
                               '--dataset-dir',str(output),'--episode-seconds','45'])
    finally:
        if output.exists():
            shutil.copy2(scene,output/'scene_reference.jpg')
            shutil.copy2(root/'scene_manifest.json',output/'scene_manifest.json')
    raise SystemExit(result.returncode)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('用户结束采集。')
    except (OSError,RuntimeError) as exc:
        print(exc,file=sys.stderr)
        raise SystemExit(1)
