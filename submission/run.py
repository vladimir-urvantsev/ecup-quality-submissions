"""Submission entry point v2.

Verdict path is byte-identical to the LB-confirmed scorer (run_base.py): same
prompt/encoder, same LoRA merge, same yes/no softmax, same thresholds and the
same override rule. What changes is robustness and the comment text:

  1. a valid CSV is written BEFORE the model loads (fallback verdicts), so a
     timeout or crash can never leave the submission empty;
  2. a wall-clock guard stops scoring when the budget is spent and keeps the
     fallback verdicts for the remaining rows;
  3. a failed batch is retried row by row instead of silently dropping rows;
  4. comments are structured, human-readable and free of probabilities; they
     come from certified text rules and never alter the verdict.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_base as base
from certificate import build_comment

T0 = time.time()
# The evaluator allows 40 minutes for ~3800 rows; stop scoring after this many
# seconds so the final CSV write always happens with margin.
BUDGET_S = int(os.environ.get("ECUP_SCORE_BUDGET_S", 31 * 60))


def default_labels(df):
    """Majority class per category: БАД is mostly 1, Легковоспламеняющиеся mostly 0."""
    return [1 if c == "БАД" else 0 for c in df["category"]]


def write_csv(df, labels, comments, path):
    results = [f"<комментарий>{base.fit_length(c)}<вердикт>{'не бан' if l == 1 else 'бан'}"
               for l, c in zip(labels, comments)]
    pd.DataFrame({"id": df["id"], "result": results}).to_csv(path, index=False)


def comments_for(df, labels, probs):
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        desc = row["description"] if pd.notna(row["description"]) else ""
        out.append(build_comment(row["category"], str(row["name"]), str(desc),
                                 labels[i], scored=probs[i] is not None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA quality classifier v2")
    ap.add_argument("-i", "--test_data_path", "--test-data-path", required=True)
    ap.add_argument("-o", "--output_path", "--output-path", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.test_data_path))
    print(f"loaded {len(df)} rows", flush=True)

    # --- stage 0: a valid submission exists from the first seconds -----------
    labels0 = default_labels(df)
    write_csv(df, labels0, comments_for(df, labels0, [None] * len(df)), args.output_path)
    print(f"stage 0: fallback CSV written ({time.time() - T0:.0f}s)", flush=True)

    # --- stage A: scoring, identical to run_base -----------------------------
    meta = {}
    meta_path = base.ADAPTER_DIR / "thresholds.json"
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
    thresholds = meta.get("thresholds", {})
    quantiles = meta.get("quantiles", {})
    max_len = int(meta.get("args", {}).get("max_len", 512))
    mode = meta.get("truncation", "tail")
    print(f"thresholds={thresholds} quantiles={quantiles} max_len={max_len} "
          f"truncation={mode}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base.BASE_MODEL, local_files_only=True)
    yes_id = tokenizer.encode("да", add_special_tokens=False)[0]
    no_id = tokenizer.encode("нет", add_special_tokens=False)[0]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base.BASE_MODEL, dtype=torch.bfloat16, local_files_only=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base.BASE_MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.to(device).eval()
    model.config.use_cache = False
    print(f"merged {base.merge_lora(model, base.ADAPTER_DIR)} LoRA modules "
          f"({time.time() - T0:.0f}s)", flush=True)

    encode = base.build_encoder(tokenizer, max_len, mode)
    pad = tokenizer.pad_token_id or 0

    def score(chunk):
        encoded = [encode(df.iloc[i]) for i in chunk]
        width = max(len(e) for e in encoded)
        input_ids = torch.tensor([[pad] * (width - len(e)) + e for e in encoded])
        attn = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in encoded])
        with torch.no_grad():
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=attn.to(device)).logits[:, -1, :]
            p = torch.softmax(
                torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1).float(),
                dim=1)[:, 0]
        return p.cpu().tolist()

    probs = [None] * len(df)
    order = sorted(range(len(df)), key=lambda i: len(str(df.iloc[i]["description"])))
    stopped = False
    for start in range(0, len(order), args.batch_size):
        if time.time() - T0 > BUDGET_S:
            print(f"  budget reached at row {start}; remaining rows keep the fallback "
                  f"verdict", file=sys.stderr, flush=True)
            stopped = True
            break
        chunk = order[start:start + args.batch_size]
        try:
            for i, value in zip(chunk, score(chunk)):
                probs[i] = value
        except Exception as exc:                # retry the batch row by row
            print(f"  batch at {start} failed ({exc}); retrying per row",
                  file=sys.stderr, flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            for i in chunk:
                try:
                    probs[i] = score([i])[0]
                except Exception as exc_row:
                    print(f"  row {df.iloc[i]['id']} failed ({exc_row}); fallback",
                          file=sys.stderr, flush=True)
        if start % (args.batch_size * 20) == 0:
            print(f"  scored {start + len(chunk)}/{len(df)} ({time.time() - T0:.0f}s)",
                  flush=True)

    # --- verdicts: same arithmetic as run_base.build_results -----------------
    cutoffs = base.resolve_thresholds(df, probs, thresholds, quantiles)
    labels = [0 if p is None else int(p >= cutoffs.get(df.iloc[i]["category"],
                                                       base.DEFAULT_THRESHOLD))
              for i, p in enumerate(probs)]
    labels, changed = base.apply_overrides(df, probs, labels)
    # rows never scored keep the category default rather than run_base's 0
    labels = [labels0[i] if p is None else labels[i] for i, p in enumerate(probs)]
    print(f"  rule overrides changed {changed} verdicts", flush=True)

    write_csv(df, labels, comments_for(df, labels, probs), args.output_path)
    scored = sum(p is not None for p in probs)
    print(f"wrote {len(df)} rows ({scored} scored, {len(df) - scored} fallback"
          f"{', budget stop' if stopped else ''}) in {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
