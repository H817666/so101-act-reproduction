"""Train one fixed-position, dual-camera ACT tube-insertion policy at a time.

The 60-episode merged dataset is read-only. Position 1 uses episodes 0-19,
position 2 uses 20-39, and position 3 uses 40-59. Two episodes per position
are held out exactly as in the earlier combined-policy experiment.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


DATASET = Path.home() / "so101-datasets" / "act_tube_slot_dual_60"
MANIFEST = Path.home() / "so101-tube-act-preflight" / "manifest.json"
OUTPUT_ROOT = Path.home() / "so101-training"
HOLDOUT = {1: [8, 18], 2: [28, 38], 3: [48, 58]}
CAMERAS = {"observation.images.front", "observation.images.wrist"}


def split_for(position, manifest, dataset_root):
    first = (position - 1) * 20
    episodes = list(range(first, first + 20))
    if len(manifest["episodes"]) != 60:
        raise RuntimeError("Expected the checked 60-episode merged dataset")
    if Path(manifest["dataset"]).name != dataset_root.name:
        raise RuntimeError("Dataset name does not match the checked manifest")
    if manifest["validation_episodes"] != [8, 18, 28, 38, 48, 58]:
        raise RuntimeError("The original validation split has changed")
    for episode in episodes:
        row = manifest["episodes"][episode]
        if row["merged_episode"] != episode or row["position"] != position or row["frames"] < 1:
            raise RuntimeError(f"Bad episode manifest entry: {episode}")
    train = [episode for episode in episodes if episode not in HOLDOUT[position]]
    return train, HOLDOUT[position]


def make_config(position, args, output, train, heldout):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    episodes = train + heldout  # The factory holds out the final 10% per task.
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=f"local/{args.dataset_root.name}",
            root=str(args.dataset_root),
            episodes=episodes,
            eval_split=0.1,
            video_backend="pyav",
        ),
        policy=ACTConfig(device="cpu" if args.check else "cuda", use_amp=True, push_to_hub=False),
        output_dir=output,
        job_name=f"act_tube_position_{position}",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        steps=args.steps,
        log_freq=200,
        env_eval_freq=0,
        eval_steps=min(5000, args.steps),
        max_eval_samples=0,
        save_checkpoint=True,
        save_freq=args.steps,  # Only a final checkpoint: ~0.6 GiB per position.
    )


def check_dataset(cfg, train_expected, heldout_expected):
    from lerobot.datasets.factory import make_train_eval_datasets

    train, validation = make_train_eval_datasets(cfg)
    if train.episodes != train_expected or validation.episodes != heldout_expected:
        raise RuntimeError(
            f"Wrong split: train={train.episodes}, validation={validation.episodes}"
        )
    if set(train.meta.camera_keys) != CAMERAS:
        raise RuntimeError(f"Wrong camera keys: {train.meta.camera_keys}")
    if len(train) < 1 or len(validation) < 1:
        raise RuntimeError("Empty training or validation split")
    for dataset in (train, validation):
        sample = dataset[0]
        for key in CAMERAS:
            if tuple(sample[key].shape[-3:]) != (3, 480, 640):
                raise RuntimeError(f"Wrong image shape for {key}: {sample[key].shape}")
        if sample["observation.state"].shape[-1] != 6 or sample["action"].shape[-1] != 6:
            raise RuntimeError("Expected six-joint state and action")
    print(
        f"Position {cfg.job_name[-1]}: {len(train.episodes)} training episodes "
        f"({len(train)} frames), {len(validation.episodes)} held-out episodes "
        f"({len(validation)} frames); both cameras decode correctly.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", type=int, choices=(1, 2, 3), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Read-only data/split check")
    mode.add_argument("--run", action="store_true", help="Train one position")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.num_workers < 1:
        parser.error("batch-size, steps, and num-workers must be positive")

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    manifest = json.loads(args.manifest.read_text())
    train, heldout = split_for(args.position, manifest, args.dataset_root)
    output = args.output_root / (
        f"act_tube_pos{args.position}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    cfg = make_config(args.position, args, output, train, heldout)
    check_dataset(cfg, train, heldout)
    if args.check:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.mem_get_info()[0] < 7 * 1024**3 and args.batch_size >= 16:
        raise RuntimeError("Less than 7 GiB GPU memory is free; stop other GPU jobs or reduce batch-size")
    args.output_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.output_root).free / 1024**3
    if free_gib < 1.8:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB of disk is free. Keep at least 1.8 GiB "
            "before starting one position; do not delete datasets or checkpoints blindly."
        )

    import draccus

    config_path = args.output_root / f"{output.name}_config.json"
    log_path = args.output_root / f"{output.name}.log"
    with draccus.config_type("json"), config_path.open("w") as f:
        draccus.dump(cfg, f)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; batch={args.batch_size}; "
        f"steps={args.steps}; disk free={free_gib:.2f} GiB\n"
        f"Only the final checkpoint is saved. Position {args.position} output: {output}\n"
        f"Log: {log_path}",
        flush=True,
    )
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "lerobot.scripts.lerobot_train", "--config_path=" + str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
