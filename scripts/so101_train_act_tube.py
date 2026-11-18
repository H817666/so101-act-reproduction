"""Local dual-camera ACT training for tube insertion; no robot connections."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path.home() / 'so101-datasets' / 'act_tube_slot_dual_60'
REPORT = Path.home() / 'so101-tube-act-preflight'
HOLDOUT = [8, 18, 28, 38, 48, 58]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--smoke', action='store_true', help='100-step GPU training, validation and checkpoint test')
    mode.add_argument('--run', action='store_true', help='Start full training')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    if not (args.check or args.smoke or args.run):
        parser.error('Specify --check, --smoke or --run')
    if args.batch_size < 1 or args.steps < 1 or args.num_workers < 0:
        parser.error('Invalid batch-size, steps or num-workers')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import draccus
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    manifest = json.loads((REPORT / 'manifest.json').read_text())
    assert manifest['validation_episodes'] == HOLDOUT and len(manifest['episodes']) == 60
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base = REPORT if args.smoke else Path.home() / 'so101-training'
    output = base / (('tube_smoke_' if args.smoke else 'act_tube_dual_') + stamp)
    steps = 100 if args.smoke else args.steps
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='local/' + ROOT.name, root=str(ROOT),
            episodes=manifest['train_episodes'] + HOLDOUT, eval_split=.1, video_backend='pyav'),
        policy=ACTConfig(device='cuda', use_amp=True, push_to_hub=False),
        output_dir=output, job_name='act_tube_slot_dual',
        batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2,
        steps=steps, log_freq=20 if args.smoke else 200,
        env_eval_freq=0, eval_steps=100 if args.smoke else 5000,
        max_eval_samples=32 if args.smoke else 0,
        save_freq=100 if args.smoke else max(1, (args.steps + 3) // 4))
    if args.check:
        train, val = make_train_eval_datasets(cfg)
        assert train.episodes == manifest['train_episodes'] and val.episodes == HOLDOUT
        assert len(train.episodes) == 54 and not set(train.episodes) & set(val.episodes)
        assert set(train.meta.camera_keys) == {'observation.images.front', 'observation.images.wrist'}
        for offset in [0, 20, 40]:
            assert sum(offset <= e < offset + 20 for e in val.episodes) == 2
        # Decode both cameras from one episode in each position in each split.
        for ds in [train, val]:
            for group in [0, 1, 2]:
                ep = next(e for e in ds.episodes if e // 20 == group)
                index = sum(manifest['episodes'][e]['frames'] for e in ds.episodes if e < ep)
                sample = ds[index]
                assert int(sample['episode_index']) == ep
                for camera in train.meta.camera_keys:
                    assert tuple(sample[camera].shape[-3:]) == (3, 480, 640)
        report = dict(train_episodes=train.episodes, validation_episodes=val.episodes,
                      train_frames=len(train), validation_frames=len(val), cameras=train.meta.camera_keys)
        (REPORT / 'split.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check NVIDIA driver before training')
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < (2 if args.smoke else 5):
        raise RuntimeError(f'Free disk {free:.2f} GiB; full training requires at least 5 GiB')
    base.mkdir(exist_ok=True)
    config_path = base / (output.name + '_config.json')
    with draccus.config_type('json'), config_path.open('w') as f:
        draccus.dump(cfg, f)
    log_path = base / (output.name + '.log')
    print(f'GPU: {torch.cuda.get_device_name(0)}\nBatch: {args.batch_size}; steps: {steps}\n'
          f'Train/validation: 54/6; each position: 18/2\n'
          f'Output: {output}\nLog: {log_path}', flush=True)
    with log_path.open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'lerobot.scripts.lerobot_train',
                                 '--config_path=' + str(config_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=os.environ.copy())
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            raise
    raise SystemExit(code)


if __name__ == '__main__':
    main()
