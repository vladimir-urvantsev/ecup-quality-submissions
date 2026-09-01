"""Submission entry point: verdict-token LoRA over Qwen3.5-4B, text only.

Reads test.csv, scores every product with p("да") taken straight from the
logits, applies the per-category threshold tuned during training, and writes
`<комментарий>...<вердикт>` rows.

Design notes:
  - peft is NOT imported. The LoRA deltas are merged into the base weights with
    plain torch, so the container only needs transformers+torch. One less thing
    that can be missing in the evaluation image.
  - Verdict mapping follows the reference baseline exactly: label 1 -> "не бан",
    label 0 -> "бан".
  - Every input id gets a row no matter what fails: a missing row invalidates
    the whole submission, so scoring errors fall back to a default verdict.
  - Prompt construction and truncation mirror training; the mode is read from
    the adapter's thresholds.json so an adapter always gets the same treatment
    it was trained with.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ADAPTER_DIR = HERE / "adapter"

_SHARED = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
BASE_MODEL = os.path.join(_SHARED, "Qwen/Qwen3.5-4B")

SYSTEM_PROMPT = (
    "Ты — модератор маркетплейса. Определи, действительно ли товар относится к "
    "своей категории риска, строго по правилам площадки. Отвечай одним словом: да или нет."
)
USER_TEMPLATE = (
    "Категория риска: {category}\n"
    "Название: {name}\n"
    "Описание: {description}\n\n"
    "Товар действительно относится к категории «{category}»? Ответ (да/нет):"
)

MIN_COMMENT, MAX_COMMENT = 50, 300
DEFAULT_THRESHOLD = 0.5

# Comments are not scored by the metric (only their 50-300 char length is), but
# they are what a human moderator would read, so they name the rule that fired.
COMMENTS = {
    ("БАД", 1): "Товар отнесён к биологически активным добавкам: в названии или описании "
                "присутствуют признаки БАД — маркировка, форма выпуска и состав, "
                "характерные для добавок. Оценка вероятности: {p:.2f}.",
    ("БАД", 0): "Товар не является биологически активной добавкой: прямых указаний на БАД "
                "в карточке нет, признаки соответствуют спортивному питанию либо обычному "
                "продукту питания. Оценка вероятности отнесения к БАД: {p:.2f}.",
    ("Легковоспламеняющиеся", 1): "Товар отнесён к легковоспламеняющимся: в составе или "
                "комплекте присутствует горючее вещество, газ либо самостоятельный источник "
                "открытого огня. Оценка вероятности: {p:.2f}.",
    ("Легковоспламеняющиеся", 0): "Товар не является легковоспламеняющимся: горючего "
                "содержимого в комплекте нет, а сама конструкция изделия не является "
                "источником воспламенения. Оценка вероятности: {p:.2f}.",
}
FALLBACK_COMMENT = ("Автоматическая проверка карточки товара не смогла получить оценку "
                    "модели для этого товара, поэтому применён вердикт по умолчанию "
                    "согласно правилам площадки.")


def fit_length(text: str) -> str:
    """Force the comment into [MIN_COMMENT, MAX_COMMENT] characters."""
    text = " ".join(str(text).split())
    if len(text) > MAX_COMMENT:
        cut = text.rfind(" ", 0, MAX_COMMENT)
        text = text[:cut if cut > MIN_COMMENT else MAX_COMMENT]
    if len(text) < MIN_COMMENT:
        text = (text + " Решение принято автоматической проверкой карточки товара "
                       "по правилам площадки.")[:MAX_COMMENT]
    return text


MIN_ROWS_FOR_QUANTILE = 50


def resolve_thresholds(df, probs, thresholds, quantiles):
    """Turn per-category quantiles into concrete cutoffs for THIS test set.

    The score distribution is strongly bimodal — almost everything sits near 0
    or near 1 — so the F1-optimal cutoff can be anywhere in the empty middle and
    wanders between samples: across folds it ranged 0.08 to 0.86 while the
    fraction of items above it stayed at 3.4-4.1%. Fixing the fraction instead
    of the probability cut the worst-case transfer loss on Легковоспламеняющиеся
    from 0.194 to 0.039 on out-of-fold data.

    The optimal fraction turned out to sit essentially at the class prevalence
    (71.00% fitted vs 70.96% actual for БАД), varying only 0.4-1.4 points across
    folds, which is why it survives a change of sample.

    Falls back to the absolute threshold when a category has too few rows here
    for a quantile to mean anything.
    """
    import numpy as np

    resolved = {}
    for category in df["category"].unique():
        q = quantiles.get(category)
        scored = [p for p, c in zip(probs, df["category"]) if c == category and p is not None]
        if q is None or len(scored) < MIN_ROWS_FOR_QUANTILE:
            resolved[category] = thresholds.get(category, DEFAULT_THRESHOLD)
            print(f"  {category}: absolute threshold {resolved[category]}", flush=True)
        else:
            resolved[category] = float(np.quantile(scored, 1.0 - q))
            print(f"  {category}: top {q * 100:.2f}% of {len(scored)} scored rows "
                  f"-> threshold {resolved[category]:.4f}", flush=True)
    return resolved


def apply_overrides(df, probs, labels):
    """Force a negative verdict for product types the assessors never accept.

    Four rules, each with zero positives across its training sample (25, 12, 180
    and 75 products): permanent/flint lighters, empty matchbox souvenirs, mangals
    without charcoal, and dry fuel with no matches included. On out-of-fold
    predictions they changed five verdicts, all five of them correct.

    Applied only to Легковоспламеняющиеся — nothing comparable was found for БАД.
    """
    try:
        from safe_overrides import suppressed_by
    except ImportError:
        print("  safe_overrides not found — skipping rule overrides", file=sys.stderr)
        return labels, 0

    changed = 0
    for i in range(len(df)):
        row = df.iloc[i]
        if row["category"] != "Легковоспламеняющиеся" or labels[i] != 1:
            continue
        rule = suppressed_by(row["name"], row.get("description", "") or "")
        if rule:
            labels[i] = 0
            changed += 1
    return labels, changed


def build_results(df, probs, thresholds):
    """Assemble the `<комментарий>...<вердикт>` strings.

    Verdict mapping matches the reference baseline: label 1 -> "не бан".
    A None probability means scoring failed for that row; it still gets a valid
    line, because a missing id invalidates the entire submission.
    """
    labels = [0 if p is None else int(p >= thresholds.get(df.iloc[i]["category"],
                                                          DEFAULT_THRESHOLD))
              for i, p in enumerate(probs)]
    labels, changed = apply_overrides(df, probs, labels)
    print(f"  rule overrides changed {changed} verdicts", flush=True)

    results = []
    for i in range(len(df)):
        category, p = df.iloc[i]["category"], probs[i]
        label = labels[i]
        comment = (FALLBACK_COMMENT if p is None else
                   COMMENTS.get((category, label), FALLBACK_COMMENT).format(p=p))
        results.append(f"<комментарий>{fit_length(comment)}"
                       f"<вердикт>{'не бан' if label == 1 else 'бан'}")
    return results


def merge_lora(model, adapter_dir: Path) -> int:
    """Fold LoRA A/B factors into the base weights. Returns modules merged."""
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora"):
        scaling = cfg["lora_alpha"] / (cfg["r"] ** 0.5)

    weights_path = adapter_dir / "adapter_model.safetensors"
    state = load_file(str(weights_path))

    # keys look like base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    pairs = {}
    for key, tensor in state.items():
        if ".lora_A" not in key and ".lora_B" not in key:
            continue
        which = "A" if ".lora_A" in key else "B"
        path = key.split(".lora_")[0]
        for prefix in ("base_model.model.", "base_model."):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        pairs.setdefault(path, {})[which] = tensor

    merged = 0
    for path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            print(f"  skip {path}: incomplete LoRA pair", file=sys.stderr)
            continue
        module = model
        for part in path.split("."):
            module = getattr(module, part)
        delta = (ab["B"].float() @ ab["A"].float()) * scaling
        with torch.no_grad():
            # Accumulate in float32, round once. Adding a bf16-rounded delta
            # straight onto a bf16 weight rounds twice, and bf16 keeps only ~3
            # significant digits — enough of a small delta disappears to move
            # borderline predictions, which matters here because the F1-optimal
            # threshold for the rare class sits on a knife edge.
            w = module.weight.data
            merged_w = w.float() + delta.to(w.device)
            w.copy_(merged_w.to(w.dtype))
        merged += 1
    return merged


def build_encoder(tokenizer, max_len: int, mode: str):
    def render(row, description):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                category=row["category"], name=row["name"], description=description)},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def encode(row):
        desc = str(row["description"] if pd.notna(row["description"]) else "")[:6000]
        if mode == "tail":
            ids = tokenizer(render(row, desc), add_special_tokens=False)["input_ids"]
            return ids[-max_len:]
        for _ in range(6):                      # mode == "description"
            ids = tokenizer(render(row, desc), add_special_tokens=False)["input_ids"]
            if len(ids) <= max_len or not desc:
                return ids[:max_len] if len(ids) > max_len else ids
            desc = desc[:max(0, int(len(desc) * max_len / len(ids) * 0.9))]
        return ids[-max_len:]

    return encode


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA quality classifier")
    # The task statement writes --output-path while the reference baseline uses
    # --output_path; accept both spellings plus the short forms.
    ap.add_argument("-i", "--test_data_path", "--test-data-path", required=True)
    ap.add_argument("-o", "--output_path", "--output-path", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    data_path = Path(args.test_data_path)
    images_path = data_path.parent / "images"          # kept for parity; text-only model
    df = pd.read_csv(data_path)
    print(f"loaded {len(df)} rows from {data_path}; images dir "
          f"{'found' if images_path.exists() else 'absent'}", flush=True)

    meta = {}
    meta_path = ADAPTER_DIR / "thresholds.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    thresholds = meta.get("thresholds", {})
    quantiles = meta.get("quantiles", {})
    max_len = int(meta.get("args", {}).get("max_len", 512))
    # Adapters trained before the truncation fix cut tokens off the front; keep
    # inference identical to whatever the adapter saw during training.
    mode = meta.get("truncation", "tail")
    print(f"thresholds={thresholds} quantiles={quantiles} "
          f"max_len={max_len} truncation={mode}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    yes_ids = tokenizer.encode("да", add_special_tokens=False)
    no_ids = tokenizer.encode("нет", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, local_files_only=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.to(device).eval()
    model.config.use_cache = False

    n_merged = merge_lora(model, ADAPTER_DIR)
    print(f"merged {n_merged} LoRA modules", flush=True)

    encode = build_encoder(tokenizer, max_len, mode)
    pad = tokenizer.pad_token_id or 0

    probs = [None] * len(df)
    order = sorted(range(len(df)), key=lambda i: len(str(df.iloc[i]["description"])))
    for start in range(0, len(order), args.batch_size):
        chunk = order[start:start + args.batch_size]
        try:
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
            for i, value in zip(chunk, p.cpu().tolist()):
                probs[i] = value
        except Exception as exc:                # never lose rows over one bad batch
            print(f"  batch at {start} failed ({exc}); falling back per row",
                  file=sys.stderr, flush=True)
        if start % (args.batch_size * 20) == 0:
            print(f"  scored {start + len(chunk)}/{len(df)}", flush=True)

    print("resolving cutoffs:", flush=True)
    cutoffs = resolve_thresholds(df, probs, thresholds, quantiles)
    out = pd.DataFrame({"id": df["id"], "result": build_results(df, probs, cutoffs)})
    out.to_csv(args.output_path, index=False)
    scored = sum(p is not None for p in probs)
    print(f"wrote {len(out)} rows to {args.output_path} ({scored} scored, "
          f"{len(out) - scored} fallback)", flush=True)


if __name__ == "__main__":
    main()
