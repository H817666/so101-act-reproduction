#!/usr/bin/env bash
set -euo pipefail

cloud_root=/root/gpufree-data/so101
python_bin=/root/gpufree-data/lerobot-env/bin/python
script="$cloud_root/so101_train_act_tube_positions.py"
dataset="$cloud_root/so101-datasets/act_tube_slot_dual_60"
manifest="$cloud_root/so101-tube-act-preflight/manifest.json"
output="$cloud_root/training"

mode="${1:-}"
position="${2:-}"
batch_size="${3:-32}"
if [[ "$mode" != --check && "$mode" != --run ]] || [[ ! "$position" =~ ^[123]$ ]] || [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: bash %s --check|--run 1|2|3 [batch_size]\n' "$0" >&2
    exit 2
fi
for file in "$python_bin" "$script" "$dataset/meta/info.json" "$manifest"; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing: %s\n' "$file" >&2
        exit 1
    fi
done

export TORCH_HOME="$cloud_root/.cache/torch"
export HF_HOME=/root/gpufree-data/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

common=(
    --position "$position"
    --batch-size "$batch_size"
    --dataset-root "$dataset"
    --manifest "$manifest"
    --output-root "$output"
)

"$python_bin" "$script" "${common[@]}" --check
if [[ "$mode" == --run ]]; then
    if [[ ! -f "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth" ]]; then
        printf 'Missing offline ResNet18 weights in TORCH_HOME\n' >&2
        exit 1
    fi
    "$python_bin" "$script" "${common[@]}" --run
fi
