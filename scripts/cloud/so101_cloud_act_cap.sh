#!/usr/bin/env bash
set -euo pipefail

cloud_data=/root/gpufree-data
robot_dir="$cloud_data/so101"
dataset_dir="$robot_dir/act_cap_tube_dual_50"
trainer="$cloud_data/lerobot-env/bin/python"
script="$robot_dir/so101_train_act_cap.py"
resnet="$cloud_data/torch-cache/hub/checkpoints/resnet18-f37072fd.pth"

case "${1:-}" in
    --check|--smoke|--run) mode=$1 ;;
    *) printf 'Usage: bash %s --check|--smoke|--run [batch_size]\n' "$0" >&2; exit 2 ;;
esac
batch_size="${2:-16}"
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    printf 'batch_size must be a positive integer\n' >&2
    exit 2
fi
for file in "$dataset_dir/meta/info.json" "$script" "$trainer"; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing: %s\n' "$file" >&2
        exit 1
    fi
done
if [[ "$mode" != --check && ! -f "$resnet" ]]; then
    printf 'Missing ResNet18 weights: %s\n' "$resnet" >&2
    exit 1
fi
export TORCH_HOME="$cloud_data/torch-cache"
export HF_HOME="$cloud_data/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
common=(--dataset-root "$dataset_dir" --report-dir "$robot_dir/preflight"
        --output-base "$robot_dir/training" --batch-size "$batch_size" --num-workers 4)

if [[ "$mode" == --run ]]; then
    "$trainer" "$script" "${common[@]}" --check
    "$trainer" "$script" "${common[@]}"
else
    "$trainer" "$script" "${common[@]}" "$mode"
fi
