#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: train_all.sh MODEL_PATH DATASET_JSONL}"
DATA="${2:?usage: train_all.sh MODEL_PATH DATASET_JSONL}"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"
for fold in 0 1 2 3 4; do
  "$PYTHON" -u src/train_lora.py \
    --model_path "$MODEL" --data "$DATA" --val_fold "$fold" \
    --output_dir "runs/fold${fold}" --epochs 2 --lr 1.5e-4 \
    --rank 16 --batch_size 2 --grad_accum 16 --max_len 1024 \
    --class_weight_power 1.0 --eval_every 200 --log_every 10 --seed 42
done

"$PYTHON" -u src/train_lora.py \
  --model_path "$MODEL" --data "$DATA" --val_fold 99 \
  --output_dir runs/full --fixed_steps 512 --lr 1.5e-4 \
  --rank 16 --batch_size 2 --grad_accum 16 --max_len 1024 \
  --class_weight_power 1.0 --log_every 10 --seed 42

