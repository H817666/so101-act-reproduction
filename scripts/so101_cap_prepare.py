"""Read-only source audit and separate curated dual-view dataset creation."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path.home() / 'so101-datasets'
REPORT = Path.home() / 'so101-cap-act-preflight'
OUT = BASE / 'act_cap_tube_dual_50'

def main():
    REPORT.mkdir(exist_ok=True)
    records = []
    for version in range(6, 10):
        root = BASE / f'cap_tube_dual_v{version}'
        info = json.loads((root/'meta/info.json').read_text())
        quality = {r['episode_index']: r['summary'] for r in map(json.loads, (root/'quality.jsonl').read_text().splitlines())}
        assert info['fps'] == 20
        for ep in range(info['total_episodes']):
            df = pd.read_parquet(root/f'data/chunk-000/file-{ep:03d}.parquet')
            n = len(df)
            assert np.array_equal(df.frame_index, np.arange(n))
            assert np.allclose(df.timestamp, np.arange(n)/20, atol=1e-5)
            action, state = (np.stack(df[k]) for k in ['action', 'observation.state'])
            assert action.shape == state.shape == (n, 6)
            assert np.isfinite(action).all() and np.isfinite(state).all()
            assert quality[ep]['accepted']
            sheet = Image.new('RGB', (960, 520), 'white')
            for row, camera in enumerate(['front', 'wrist']):
                cap = cv2.VideoCapture(str(root/f'videos/observation.images.{camera}/chunk-000/file-{ep:03d}.mp4'))
                count = 0
                selected = [0, n//2, n-1]
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    assert frame.shape == (480, 640, 3)
                    if count in selected:
                        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320,240))
                        sheet.paste(im, (selected.index(count)*320,row*260+20))
                    count += 1
                cap.release()
                assert count == n, (version,ep,camera,count,n)
                ImageDraw.Draw(sheet).text((0,row*260),f'v{version} episode {ep+1} ({camera}) start / middle / end',fill='black')
            sheet.save(REPORT/f'v{version}_ep{ep+1:02d}.jpg')
            excluded = version == 8 and ep == 13
            records.append(dict(source=str(root),source_episode=ep,excluded=excluded,
                frames=n,quality=quality[ep],max_action_step_first5=float(np.abs(np.diff(action[:,:5],axis=0)).max())))
        print('Audited',root.name,flush=True)
    kept = [r for r in records if not r['excluded']]
    assert len(kept) == 50
    for index, record in enumerate(kept):
        record['merged_episode'] = index
    (REPORT/'audit.json').write_text(json.dumps(records,indent=2))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.aggregate import aggregate_datasets
    clean = REPORT/'v8_filtered'
    if not clean.exists():
        src = LeRobotDataset('local/cap_tube_dual_v8',root=BASE/'cap_tube_dual_v8',video_backend='pyav')
        delete_episodes(src,[13],output_dir=clean,repo_id='local/cap_tube_v8_filtered')
    roots = [BASE/'cap_tube_dual_v6',BASE/'cap_tube_dual_v7',clean,BASE/'cap_tube_dual_v9']
    if not OUT.exists():
        aggregate_datasets(repo_ids=['local/'+r.name for r in roots],roots=roots,
            aggr_repo_id='local/act_cap_tube_dual_50',aggr_root=OUT,
            concatenate_videos=False,concatenate_data=False)
    ds = LeRobotDataset('local/act_cap_tube_dual_50',root=OUT,video_backend='pyav')
    assert ds.num_episodes == 50 and len(ds) == sum(r['frames'] for r in kept)
    for r in kept:
        original = pd.read_parquet(Path(r['source'])/f"data/chunk-000/file-{r['source_episode']:03d}.parquet")
        merged = pd.read_parquet(OUT/f"data/chunk-000/file-{r['merged_episode']:03d}.parquet")
        for k in ['action','observation.state','timestamp','frame_index']:
            assert np.array_equal(np.stack(original[k]),np.stack(merged[k])), (r,k)
    print('PASS:',ds.num_episodes,'episodes',len(ds),'frames',ds.meta.camera_keys,flush=True)

if __name__ == '__main__':
    main()
