#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: evaluate_oof.sh MODEL_PATH DATASET_JSONL}"
DATA="${2:?usage: evaluate_oof.sh MODEL_PATH DATASET_JSONL}"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"
mkdir -p preds
for fold in 0 1 2 3 4; do
  "$PYTHON" -u src/evaluate_adapter.py \
    --model_path "$MODEL" --adapter "runs/fold${fold}/best" \
    --data "$DATA" --training_data "$DATA" --val_fold "$fold" \
    --allow_unverified_adapter_fold --out "preds/fold${fold}.csv"
done

"$PYTHON" src/evaluate_competition_metric.py 'preds/fold*.csv' \
  --threshold 'БАД=0.2123' --threshold 'Легковоспламеняющиеся=0.717'

