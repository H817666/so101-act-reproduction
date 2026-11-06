"""Merge audited V2 recordings without changing sources or touching hardware."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home()/'so101-datasets-v2')
    parser.add_argument('--report-dir', type=Path, default=Path.home()/'so101-v2-audit')
    args = parser.parse_args()
    base = args.root.expanduser().resolve()
    out = base/'act_pick_place_v2_60'
    report = args.report_dir.expanduser().resolve()
    entries = json.loads((report / 'report.json').read_text())
    assert len(entries) == 60 and all(e['quality']['accepted'] for e in entries)
    roots = list(dict.fromkeys(Path(e['root']) for e in entries))
    configs = [json.loads((r/'collection_config.json').read_text()) for r in roots]
    assert all(c == configs[0] for c in configs), 'Collection/calibration mismatch'
    for root in roots:
        info = json.loads((root/'meta/info.json').read_text())
        rows = [e for e in entries if Path(e['root']) == root]
        assert len(rows) == info['total_episodes']
        assert sum(e['frames'] for e in rows) == info['total_frames']
    if not out.exists():
        aggregate_datasets(repo_ids=['local/'+r.name for r in roots], roots=roots,
                           aggr_repo_id='local/act_pick_place_v2_60', aggr_root=out,
                           concatenate_videos=False, concatenate_data=False)
    ds = LeRobotDataset('local/act_pick_place_v2_60', root=out, video_backend='pyav')
    assert ds.num_episodes == 60 and len(ds) == sum(e['frames'] for e in entries)
    offset = 0
    for ep, entry in enumerate(entries):
        root = Path(entry['root']); local = entry['local_episode']
        src = pq.read_table(root/f'data/chunk-000/file-{local:03}.parquet')
        dst = pq.read_table(out/ds.meta.get_data_file_path(ep))
        for key in ('action', 'observation.state', 'timestamp', 'frame_index'):
            assert src[key].to_pylist() == dst[key].to_pylist(), (ep, key)
        src_video = root/f'videos/observation.images.front/chunk-000/file-{local:03}.mp4'
        dst_video = out/ds.meta.get_video_file_path(ep, 'observation.images.front')
        assert hashlib.sha256(src_video.read_bytes()).digest() == hashlib.sha256(dst_video.read_bytes()).digest()
        for idx in (offset, offset+entry['frames']-1):
            sample = ds[idx]
            assert int(sample['episode_index']) == ep
            assert tuple(sample['observation.images.front'].shape) == (3,480,640)
        offset += entry['frames']
    for name in ('scene_reference.jpg','scene_manifest.json'):
        shutil.copy2(base/name, out/name)
    shutil.copy2(roots[0]/'collection_config.json', out/'collection_config.json')
    (out/'source_manifest.json').write_text(json.dumps(entries, indent=2))
    result = dict(episodes=ds.num_episodes, frames=len(ds), source_count=len(roots),
                  all_video_sha256_equal=True, all_action_state_timestamp_equal=True,
                  episode_boundary_reads=120, root=str(out))
    (report/'merge.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
