"""Scores a saved LoRA adapter on a validation fold and dumps per-example
probabilities, so every experiment can be compared on identical footing.

  python evaluate_adapter.py --adapter runs/exp1_base/best --val_fold 0 \
      --model_path models/Qwen3.5-4B --out preds/exp1_fold0.csv

Why a separate script instead of trusting the training logs:
  - runs made before a metric change are not comparable with later ones;
  - a --category run only ever reports its own category;
  - saved probabilities enable offline threshold fitting, ensembling and error
    analysis without re-running the model.

It merges the adapter with the same plain-torch code the submission uses, so a
successful run here also proves that path works.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_lora import (
    POLICY_QUERY_FIELD,
    SYSTEM_PROMPT,
    TRAINING_CONFIG_SCHEMA,
    encode_description_truncated,
    hash_training_config,
    load_examples,
    policy_query_data_contract,
    render_classifier_prompt,
    resolve_policy_query_field,
    tune_threshold_macro_f1,
)
from backbone_compat import (
    BACKBONE_PROFILES,
    GEMMA4_E4B_PROFILE,
    LEGACY_QWEN_PROFILE,
    BackboneContractError,
    resolve_backbone_profile,
)


TEXT_PRED_SCHEMA = "text-oof-predictions-v2"


def merge_lora(model, adapter_dir: Path, *, allowed_paths=None) -> int:
    """Fold LoRA A/B factors into base weights — same logic as submit_lora/run.py."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora"):
        scaling = cfg["lora_alpha"] / (cfg["r"] ** 0.5)

    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
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

    if allowed_paths is not None:
        expected = set(map(str, allowed_paths))
        actual = set(pairs)
        incomplete = {
            path: sorted(value)
            for path, value in pairs.items()
            if set(value) != {"A", "B"}
        }
        if incomplete:
            raise ValueError(f"incomplete Gemma LoRA A/B pairs: {incomplete}")
        if actual != expected:
            raise ValueError(
                "Gemma adapter merge scope mismatch: "
                f"missing={sorted(expected - actual)[:10]} "
                f"extra={sorted(actual - expected)[:10]}"
            )

    merged = 0
    for path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        module = model
        for part in path.split("."):
            module = getattr(module, part)
        delta = (ab["B"].float() @ ab["A"].float()) * scaling
        with torch.no_grad():
            # float32 accumulation, single rounding — see submit_lora/run.py
            w = module.weight.data
            w.copy_((w.float() + delta.to(w.device)).to(w.dtype))
        merged += 1
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model_path", default="models/Qwen3.5-4B")
    ap.add_argument(
        "--backbone_profile",
        choices=BACKBONE_PROFILES,
        default=None,
        help="normally inferred from checkpoint metadata",
    )
    ap.add_argument(
        "--backbone_preflight",
        default=None,
        help="hash-bound Gemma preflight used by the training run",
    )
    ap.add_argument("--data", default="data/lora_dataset.jsonl")
    ap.add_argument(
        "--training_data", default=None,
        help=("manifest used to train the adapter; required when --data is a "
              "row-level evaluation manifest such as multimodal_dataset.jsonl"),
    )
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--category", default=None,
                    help="restrict evaluation to one category")
    ap.add_argument(
        "--policy_query_field",
        choices=(POLICY_QUERY_FIELD,),
        default=None,
        help=("normally inferred from checkpoint metadata; an explicit value "
              "must be compatible with the stored training contract"),
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=None, help="csv for per-example probabilities")
    ap.add_argument("--allow_unverified_adapter_fold", action="store_true",
                    help="diagnostic only: score when checkpoint val_fold is absent/mismatched")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter)
    meta = {}
    if (adapter_dir / "thresholds.json").exists():
        meta = json.loads((adapter_dir / "thresholds.json").read_text())
    checkpoint_fold = meta.get("outer_fold", meta.get("val_fold"))
    outer_fold_untouched = bool(meta.get("outer_fold_untouched", False))
    checkpoint_selection_fold = meta.get(
        "checkpoint_selection_fold", meta.get("val_fold"))
    current_data_sha256 = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    training_data_path = Path(args.training_data or args.data)
    current_training_data_sha256 = hashlib.sha256(
        training_data_path.read_bytes()
    ).hexdigest()
    checkpoint_data_sha256 = meta.get("data_sha256")
    stored_training_config = meta.get("training_config")
    stored_training_config_sha256 = str(
        meta.get("training_config_sha256", "")
    )
    try:
        backbone_profile = resolve_backbone_profile(
            args.backbone_profile,
            checkpoint_args=meta.get("args", {}),
            training_config=stored_training_config,
        )
    except BackboneContractError as exc:
        raise SystemExit(f"incompatible backbone evaluation contract: {exc}")
    backbone_contract_verified = True
    gemma_preflight = None
    if backbone_profile == LEGACY_QWEN_PROFILE:
        if args.backbone_preflight:
            raise SystemExit(
                "--backbone_preflight is forbidden for the legacy Qwen profile"
            )
    else:
        from backbone_compat import sha256_file, validate_preflight_file

        if backbone_profile != GEMMA4_E4B_PROFILE or not args.backbone_preflight:
            raise SystemExit(
                "Gemma evaluation requires --backbone_preflight from training"
            )
        try:
            gemma_preflight = validate_preflight_file(
                args.backbone_preflight,
                args.model_path,
                backbone_profile,
                require_weight_hashes=True,
            )
        except (OSError, BackboneContractError) as exc:
            raise SystemExit(f"invalid Gemma backbone preflight: {exc}")
        runtime = meta.get("backbone_runtime_provenance")
        backbone_contract_verified = bool(
            isinstance(stored_training_config, dict)
            and stored_training_config.get("backbone_profile") == backbone_profile
            and stored_training_config.get("backbone_preflight_sha256")
            == sha256_file(args.backbone_preflight)
            and stored_training_config.get("backbone_compat_code_sha256")
            == sha256_file(Path(__file__).with_name("backbone_compat.py"))
            and isinstance(runtime, dict)
            and runtime.get("profile") == backbone_profile
            and runtime.get("preflight_sha256")
            == sha256_file(args.backbone_preflight)
        )
    try:
        policy_query_field = resolve_policy_query_field(
            args.policy_query_field,
            checkpoint_args=meta.get("args", {}),
            training_config=stored_training_config,
        )
    except ValueError as exc:
        raise SystemExit(f"incompatible policy-query evaluation contract: {exc}")
    policy_query_contract_verified = True
    if policy_query_field is not None:
        try:
            actual_query_contract = policy_query_data_contract(
                training_data_path, policy_query_field
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid policy-query training manifest: {exc}")
        policy_query_contract_verified = bool(
            isinstance(stored_training_config, dict)
            and stored_training_config.get("policy_query_field")
            == policy_query_field
            and stored_training_config.get("policy_query_contract")
            == actual_query_contract
        )
    training_config_verified = bool(
        isinstance(stored_training_config, dict) and
        stored_training_config.get("schema_version") == TRAINING_CONFIG_SCHEMA and
        hash_training_config(stored_training_config)
        == stored_training_config_sha256 and
        stored_training_config.get("training_data_sha256")
        == current_training_data_sha256 and
        stored_training_config.get("model_path")
        == str(Path(args.model_path).resolve())
        and policy_query_contract_verified
        and backbone_contract_verified
    )
    fold_verified = (checkpoint_fold is not None and
                     int(checkpoint_fold) == args.val_fold and
                     outer_fold_untouched and
                     checkpoint_selection_fold is None and
                     checkpoint_data_sha256 == current_training_data_sha256 and
                     training_config_verified)
    if not fold_verified and not args.allow_unverified_adapter_fold:
        raise SystemExit(
            f"adapter provenance failed: outer_fold={checkpoint_fold!r}, "
            f"outer_fold_untouched={outer_fold_untouched}, requested "
            f"--val_fold={args.val_fold}, data hash match="
            f"{checkpoint_data_sha256 == current_training_data_sha256}, "
            f"training config verified={training_config_verified}. "
            "Refusing a checkpoint "
            "selected in-fold or trained on another manifest; "
            "pass --training_data for row-level evaluation, or use "
            "--allow_unverified_adapter_fold only for diagnostics."
        )
    max_len = int(meta.get("args", {}).get("max_len", 512))
    mode = meta.get("truncation", "tail")
    print(f"adapter={adapter_dir} max_len={max_len} truncation={mode}")

    _, val = load_examples(args.data, args.val_fold, args.category)
    if policy_query_field is not None:
        missing = [
            int(rec["id"])
            for rec in val
            if not isinstance(rec.get(policy_query_field), str)
            or not rec[policy_query_field].strip()
        ]
        if missing:
            raise SystemExit(
                f"evaluation manifest is missing non-empty {policy_query_field!r} "
                f"for {len(missing)} rows; first ids={missing[:20]}"
            )
    print(f"val examples: {len(val)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    expected_gemma_targets = None
    if backbone_profile == LEGACY_QWEN_PROFILE:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        yes_ids = tokenizer.encode("да", add_special_tokens=False)
        no_ids = tokenizer.encode("нет", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids == no_ids:
            raise SystemExit(
                f"verdict tokens must be distinct singletons, got {yes_ids}/{no_ids}"
            )
        yes_id, no_id = yes_ids[0], no_ids[0]
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, dtype=torch.bfloat16, local_files_only=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
        model = model.to(device).eval()
        model.config.use_cache = False
    else:
        from backbone_compat import (
            disable_model_cache,
            load_model_for_profile,
            load_tokenizer_for_profile,
            resolve_gemma4_lora_targets,
            validate_adapter_scope,
            validate_tokenizer_against_preflight,
        )

        tokenizer, verdict_contract = load_tokenizer_for_profile(
            args.model_path, backbone_profile
        )
        try:
            validate_tokenizer_against_preflight(
                verdict_contract, gemma_preflight
            )
        except BackboneContractError as exc:
            raise SystemExit(f"Gemma tokenizer provenance failed: {exc}")
        yes_id = verdict_contract["verdicts"]["yes"]["token_id"]
        no_id = verdict_contract["verdicts"]["no"]["token_id"]
        model = load_model_for_profile(
            args.model_path, backbone_profile, dtype=torch.bfloat16
        ).to(device).eval()
        disable_model_cache(model, backbone_profile)
        expected_gemma_targets, target_audit = resolve_gemma4_lora_targets(model)
        runtime = meta["backbone_runtime_provenance"]
        if (
            target_audit["target_paths_sha256"]
            != runtime.get("target_paths_sha256")
        ):
            raise SystemExit("Gemma runtime target inventory drifted since training")
        try:
            validate_adapter_scope(adapter_dir, expected_gemma_targets)
        except BackboneContractError as exc:
            raise SystemExit(f"invalid Gemma adapter scope: {exc}")
    print(
        f"merged {merge_lora(model, adapter_dir, allowed_paths=expected_gemma_targets)} "
        "LoRA modules"
    )

    def render(rec, description):
        return render_classifier_prompt(
            tokenizer,
            rec,
            description,
            policy_query_field=policy_query_field,
        )

    def encode(rec):
        desc = rec["description"][:6000]
        if mode == "tail":
            if policy_query_field is not None:
                raise ValueError(
                    "policy-query checkpoints require description truncation; "
                    "tail truncation could remove the instruction"
                )
            return tokenizer(render(rec, desc),
                             add_special_tokens=False)["input_ids"][-max_len:]
        if policy_query_field is None:
            # Historical evaluator path, intentionally byte/token equivalent
            # for ordinary checkpoints.
            for _ in range(6):
                ids = tokenizer(
                    render(rec, desc), add_special_tokens=False
                )["input_ids"]
                if len(ids) <= max_len or not desc:
                    return ids[:max_len] if len(ids) > max_len else ids
                desc = desc[:max(0, int(len(desc) * max_len / len(ids) * 0.9))]
            return ids[-max_len:]
        return encode_description_truncated(
            tokenizer,
            rec,
            render,
            max_len,
            policy_query_field=policy_query_field,
        )

    pad = tokenizer.pad_token_id or 0
    t0 = time.time()
    probs, margins = [], []
    for start in range(0, len(val), args.batch_size):
        chunk = val[start:start + args.batch_size]
        encoded = [encode(r) for r in chunk]
        width = max(len(e) for e in encoded)
        ids = torch.tensor([[pad] * (width - len(e)) + e for e in encoded])
        attn = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in encoded])
        with torch.no_grad():
            logits = model(input_ids=ids.to(device),
                           attention_mask=attn.to(device)).logits[:, -1, :]
            ly, ln = logits[:, yes_id].float(), logits[:, no_id].float()
            p = torch.softmax(torch.stack([ly, ln], dim=1), dim=1)[:, 0]
        probs += p.cpu().tolist()
        # Raw margin, kept because the sigmoid of it is on an arbitrary
        # per-adapter scale: two models can rank identically yet need thresholds
        # of .12 and .85. Calibration works on z, not on p.
        margins += (ly - ln).cpu().tolist()
        if start % (args.batch_size * 20) == 0:
            print(f"  {start + len(chunk)}/{len(val)}", flush=True)
    print(f"scored in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / max(1, len(val)) * 1000:.0f} ms/example)")

    labels = [r["label"] for r in val]
    cats = [r["category"] for r in val]
    honest, optimistic, thresholds, per_cat = tune_threshold_macro_f1(labels, probs, cats)

    print(f"\nmacro F1 = {honest:.4f} (honest) | {optimistic:.4f} (optimistic)")
    for c, (h, o, n) in per_cat.items():
        print(f"  {c:24s} honest {h:.4f} | optimistic {o:.4f} "
              f"| threshold {thresholds[c]} | {n} positives")

    # Class weighting changes the optimum logit scale by construction.  The raw
    # restricted-softmax p_yes is therefore a decision score, not automatically
    # an estimate of the natural posterior P(y=1|x).  ROC-AUC measures global
    # ordering; PR-AUC and local boundary errors can still move when ROC-AUC is
    # nearly flat, so no single threshold-independent metric proves equivalence.
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    labels_a, probs_a, cats_a = np.array(labels), np.array(probs), np.array(cats)
    print("\nthreshold-independent ranking diagnostics:")
    for c in sorted(set(cats)):
        m = cats_a == c
        if len(set(labels_a[m])) < 2:
            continue
        print(f"  {c:24s} AUC {roc_auc_score(labels_a[m], probs_a[m]):.4f} "
              f"| PR-AUC {average_precision_score(labels_a[m], probs_a[m]):.4f} "
              f"| raw-score Brier {brier_score_loss(labels_a[m], probs_a[m]):.4f}")

    print("\nraw-score reliability (not posterior calibration under weighted CE):")
    for c in sorted(set(cats)):
        m = cats_a == c
        row = []
        for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
            b = m & (probs_a >= lo) & (probs_a < hi)
            if b.sum() >= 5:
                row.append(f"[{lo:.1f},{hi:.1f}) n={int(b.sum())} "
                           f"заявлено {probs_a[b].mean():.2f} факт {labels_a[b].mean():.2f}")
        print(f"  {c}")
        for r in row:
            print(f"     {r}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        output_columns = {
            "id": [r["id"] for r in val],
            "category": cats,
            "label": labels,
            "fold": [r["fold"] for r in val],
            "p_yes": probs,
            "z": margins,
            "text_pred_schema": TEXT_PRED_SCHEMA,
            "adapter_path": str(adapter_dir.resolve()),
            "adapter_val_fold": checkpoint_fold,
            "adapter_fold_verified": int(fold_verified),
            "outer_fold_untouched": int(outer_fold_untouched),
            "checkpoint_selection_fold": checkpoint_selection_fold,
            "checkpoint_step": meta.get("step"),
            "data_sha256": current_data_sha256,
            "adapter_training_data_sha256": current_training_data_sha256,
            "adapter_training_config_sha256": stored_training_config_sha256,
            "model_path": str(Path(args.model_path).resolve()),
            "policy_query_field": policy_query_field,
            "policy_query_id": [
                r.get("policy_query_id") if policy_query_field else None for r in val
            ],
            "policy_query_variant": [
                r.get("policy_query_variant") if policy_query_field else None for r in val
            ],
            "policy_query_view": [
                r.get("policy_query_view") if policy_query_field else None for r in val
            ],
        }
        if backbone_profile != LEGACY_QWEN_PROFILE:
            output_columns.update({
                "backbone_profile": backbone_profile,
                "backbone_preflight_sha256": stored_training_config.get(
                    "backbone_preflight_sha256"
                ),
                "backbone_target_paths_sha256": meta[
                    "backbone_runtime_provenance"
                ]["target_paths_sha256"],
            })
        pd.DataFrame(output_columns).to_csv(out, index=False)
        print(f"\nper-example probabilities -> {out}")


if __name__ == "__main__":
    main()
