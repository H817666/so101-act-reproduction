"""Offline ACT training entrypoint. Never connects to robot hardware."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime

HOLDOUT = [9, 19, 29, 39, 49]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='30-step pipeline check only')
    p.add_argument('--check', action='store_true', help='Verify episode split without training')
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--v2', action='store_true', help='New fixed-scene 60-episode dataset; train from scratch')
    p.add_argument('--dataset-root', type=Path, help='Merged LeRobot dataset directory')
    p.add_argument('--output-root', type=Path, default=Path.home()/'so101-training')
    args = p.parse_args()
    total = 60 if args.v2 else 50
    holdout = [8, 18, 28, 38, 48, 58] if args.v2 else HOLDOUT
    repo = 'local/act_pick_place_v2_60' if args.v2 else 'local/act_pick_place_50'
    root = args.dataset_root or (Path.home()/'so101-datasets-v2'/'act_pick_place_v2_60' if args.v2
                                 else Path.home()/'so101-datasets'/'act_pick_place_50')
    root = root.expanduser().resolve()
    if not (root/'meta'/'info.json').is_file():
        p.error(f'未找到 LeRobot 数据集：{root}')
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if not args.check:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing accidental CPU training.')
        args.output_root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.output_root).free < 6 * 1024**3:
            raise RuntimeError('Less than 6 GiB free. Free disk space before training.')
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    import draccus

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base = args.output_root.expanduser() / ('preflight' if args.smoke else 'runs')
    output = base / ('smoke_' + stamp if args.smoke else 'act_' + stamp)
    if args.v2:
        output = output.with_name(output.name + '_v2')
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo,
            root=root,
            episodes=[i for i in range(total) if i not in holdout] + holdout,
            eval_split=.1, video_backend='pyav'),
        policy=ACTConfig(device='cuda', use_amp=True, push_to_hub=False),
        output_dir=output, job_name='so101_act_pick_place',
        batch_size=args.batch_size, num_workers=4, prefetch_factor=2,
        steps=30 if args.smoke else 100000,
        log_freq=10 if args.smoke else 200,
        env_eval_freq=0, eval_steps=30 if args.smoke else 5000,
        max_eval_samples=96 if args.smoke else 0,
        save_freq=30 if args.smoke else 25000,
    )
    if args.check:
        train, val = make_train_eval_datasets(cfg)
        assert val.episodes == holdout, val.episodes
        assert len(train.episodes) == total-len(holdout) and not set(train.episodes) & set(val.episodes)
        report = dict(train_episodes=train.episodes, validation_episodes=val.episodes,
                      train_frames=len(train), validation_frames=len(val))
        report_path = args.output_root.expanduser() / ('split_v2.json' if args.v2 else 'split.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return
    base.mkdir(parents=True, exist_ok=True)
    config_path = base / (output.name + '_config.json')
    with draccus.config_type('json'):
        with config_path.open('w') as f:
            draccus.dump(cfg, f)
    print('Output:', output, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1')
    log_path = base / (output.name + '.log')
    print('Log:', log_path, flush=True)
    with log_path.open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'lerobot.scripts.lerobot_train',
                                 '--config_path=' + str(config_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
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
