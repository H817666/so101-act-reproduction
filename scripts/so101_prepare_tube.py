"""Prepare the three audited tube positions as a separate LeRobot dataset."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BASE = Path.home() / 'so101-datasets'
OUT = BASE / 'act_tube_slot_dual_60'
REPORT = Path.home() / 'so101-tube-act-preflight'
VERSIONS = (4, 5, 7)
HOLDOUT = [8, 18, 28, 38, 48, 58]


def main():
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    roots = [BASE / f'tube_slot_3pos_dual_v{v}' for v in VERSIONS]
    records = []
    for position, root in enumerate(roots, 1):
        info = json.loads((root / 'meta/info.json').read_text())
        audit = json.loads((Path.home() / 'so101-audits' / root.name / 'report.json').read_text())
        assert info['total_episodes'] == len(audit['episodes']) == 20
        assert info['fps'] == 20
        for e, item in enumerate(audit['episodes']):
            assert item['episode_ui'] == e + 1
            assert item['summary']['accepted'] and all(item['checks'].values())
            assert all(v['matches'] for v in item['video'].values())
            records.append(dict(source=str(root), source_episode=e, position=position,
                                merged_episode=len(records), frames=item['frames']))
    REPORT.mkdir(exist_ok=True)
    if not OUT.exists():
        aggregate_datasets(repo_ids=['local/' + r.name for r in roots], roots=roots,
                           aggr_repo_id='local/' + OUT.name, aggr_root=OUT,
                           concatenate_videos=False, concatenate_data=False)
    ds = LeRobotDataset('local/' + OUT.name, root=OUT, video_backend='pyav')
    assert ds.num_episodes == 60 and len(ds) == sum(r['frames'] for r in records)
    assert set(ds.meta.camera_keys) == {'observation.images.front', 'observation.images.wrist'}
    for r in records:
        src = pq.read_table(Path(r['source']) / f"data/chunk-000/file-{r['source_episode']:03d}.parquet").to_pydict()
        dst = pq.read_table(OUT / f"data/chunk-000/file-{r['merged_episode']:03d}.parquet").to_pydict()
        for key in ['action', 'observation.state', 'timestamp', 'frame_index']:
            assert np.array_equal(np.asarray(src[key]), np.asarray(dst[key])), (r, key)
        assert set(dst['episode_index']) == {r['merged_episode']}
    manifest = dict(dataset=str(OUT), frames=len(ds), episodes=records,
                    validation_episodes=HOLDOUT,
                    train_episodes=[i for i in range(60) if i not in HOLDOUT])
    (REPORT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'PASS: 60 episodes, {len(ds)} frames; positions 20/20/20; {OUT}')


if __name__ == '__main__':
    main()
