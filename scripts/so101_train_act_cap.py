"""Dual-camera ACT cap placement training. Does not access robot hardware."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path.home() / 'so101-datasets' / 'act_cap_tube_dual_50'
REPORT = Path.home() / 'so101-cap-act-preflight'
HOLDOUT = [4, 14, 24, 39, 49]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--steps',type=int,default=100000)
    parser.add_argument('--dataset-root',type=Path,default=ROOT)
    parser.add_argument('--output-base',type=Path,default=None)
    parser.add_argument('--report-dir',type=Path,default=REPORT)
    parser.add_argument('--num-workers',type=int,default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.num_workers < 0:
        parser.error('batch-size and steps must be positive; num-workers must be nonnegative')
    dataset_root = args.dataset_root.resolve()
    if not (dataset_root/'meta/info.json').is_file():
        parser.error(f'dataset not found: {dataset_root}')
    import draccus
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base = args.output_base or (args.report_dir if args.smoke else Path.home() / 'so101-training')
    output = base / ('cap_smoke_' + stamp if args.smoke else 'act_cap_dual_' + stamp)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='local/act_cap_tube_dual_50',root=str(dataset_root),
            episodes=[i for i in range(50) if i not in HOLDOUT]+HOLDOUT,
            eval_split=.1,video_backend='pyav'),
        policy=ACTConfig(device='cuda',use_amp=True,push_to_hub=False),
        output_dir=output,job_name='act_cap_tube_dual',
        batch_size=args.batch_size,num_workers=args.num_workers,prefetch_factor=2,
        steps=30 if args.smoke else args.steps,log_freq=10 if args.smoke else 200,
        env_eval_freq=0,eval_steps=30 if args.smoke else 5000,
        max_eval_samples=32 if args.smoke else 0,
        save_freq=30 if args.smoke else max(1,(args.steps+3)//4))
    if args.check:
        train,val = make_train_eval_datasets(cfg)
        assert len(train.episodes)==45 and val.episodes==HOLDOUT
        assert not set(train.episodes)&set(val.episodes)
        assert set(train.meta.camera_keys)=={'observation.images.front','observation.images.wrist'}
        report=dict(train_episodes=train.episodes,validation_episodes=val.episodes,
                    train_frames=len(train),validation_frames=len(val),cameras=train.meta.camera_keys)
        args.report_dir.mkdir(parents=True,exist_ok=True)
        (args.report_dir/'split.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU training')
    if shutil.disk_usage(dataset_root).free < (2 if args.smoke else 5)*1024**3:
        raise RuntimeError('Insufficient free space: need 5 GiB for full training, 2 GiB for smoke')
    base.mkdir(exist_ok=True)
    config_path=base/(output.name+'_config.json')
    with draccus.config_type('json'),config_path.open('w') as f:
        draccus.dump(cfg,f)
    log_path=base/(output.name+'.log')
    print('Output:',output,'\nLog:',log_path,flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with log_path.open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','lerobot.scripts.lerobot_train','--config_path='+str(config_path)],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
        try:
            for line in proc.stdout:
                print(line,end='',flush=True)
                log.write(line)
                log.flush()
            code=proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            raise
    raise SystemExit(code)

if __name__ == '__main__':
    main()
