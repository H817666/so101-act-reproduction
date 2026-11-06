"""Read-only round-two dataset audit; diagnostic artifacts only."""
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path.home()/'so101-datasets-v2')
parser.add_argument('--report-dir',type=Path,default=Path.home()/'so101-v2-audit')
args=parser.parse_args()
ROOT=args.root.expanduser().resolve()
OUT=args.report_dir.expanduser().resolve();OUT.mkdir(parents=True,exist_ok=True)
report=[]; sheets=[]
for root in sorted(ROOT.glob('pick_place_*')):
    info=json.loads((root/'meta/info.json').read_text())
    if info['total_episodes'] == 0:
        assert info['total_frames'] == 0
        print('SKIP EMPTY',root.name,flush=True)
        continue
    quality={q['episode_index']:q['summary'] for q in map(json.loads,(root/'quality.jsonl').read_text().splitlines())}
    print(root.name,info['total_episodes'],info['total_frames'],flush=True)
    for path in sorted((root/'data').rglob('*.parquet')):
        d=pq.read_table(path).to_pydict()
        a=np.asarray(d['action']);s=np.asarray(d['observation.state']);n=len(a)
        e=int(d['episode_index'][0]);assert len(set(d['episode_index']))==1
        assert np.isfinite(a).all() and np.isfinite(s).all()
        assert np.array_equal(d['frame_index'],np.arange(n))
        assert np.allclose(d['timestamp'],np.arange(n)/20,atol=1e-4)
        video=root/'videos/observation.images.front'/path.parent.name/(path.stem+'.mp4')
        cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        assert abs(cap.get(cv2.CAP_PROP_FPS)-20)<.01
        picks=np.linspace(0,n-1,7).astype(int).tolist();selected=[];count=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert bgr.shape==(480,640,3)
            if count in picks:
                frame=cv2.resize(bgr,(256,192))
                cv2.putText(frame,f'E{len(report)+1} {count/20:.1f}s',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
                selected.append(frame)
            count+=1
        cap.release();assert count==n
        q=quality[e];assert q['frames']==n
        item={'episode':len(report)+1,'root':str(root),'local_episode':e,'frames':n,'duration_s':n/20,
              'quality':q,'start':s[0].tolist(),'end':s[-1].tolist(),
              'gripper_state_minmax':[float(s[:,5].min()),float(s[:,5].max())],
              'gripper_action_minmax':[float(a[:,5].min()),float(a[:,5].max())],
              'tracking_p95':np.quantile(abs(a[:-1]-s[1:]),.95,axis=0).tolist(),
              'command_speed_max':(abs(np.diff(a,axis=0))*20).max(0).tolist()}
        report.append(item);sheets.append(np.concatenate(selected,axis=1))
        print(json.dumps({k:v for k,v in item.items() if k not in ['root','quality']},ensure_ascii=False),flush=True)
for i in range(0,len(sheets),5):
    cv2.imwrite(str(OUT/f'episodes_{i+1}_{min(i+5,len(sheets))}.jpg'),np.concatenate(sheets[i:i+5],axis=0))
(OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print('TOTAL',len(report),sum(r['frames'] for r in report),'all accepted',all(r['quality']['accepted'] for r in report))
print('sampling_hz',min(r['quality']['effective_hz'] for r in report),max(r['quality']['effective_hz'] for r in report))
print('max_gap',max(r['quality']['max_gap_s'] for r in report),'max_camera_age',max(r['quality']['max_camera_age_s'] for r in report))
