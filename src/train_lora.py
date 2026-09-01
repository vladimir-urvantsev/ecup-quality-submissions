"""LoRA fine-tuning: verdict-token classification on top of Qwen3.5-4B (text-only).

The model is trained to answer a single token — "да" (label=1) or "нет" (label=0) —
to the question "does this product genuinely belong to its risk category". At
inference p(да) is read from the logits directly (no generation / parsing).

Key features:
  - grouped working folds from lora_dataset.jsonl (they block declared product
    families, but are not claimed semantic/image-family leakage-free);
  - per-example weights (duplicates + hard-example boost) and per-category
    class weights (Легковоспламеняющиеся has only ~143 positives);
  - optional soft labels (--soft_labels): CE against assessor agreement
    p instead of hard 0/1 — the "model the assessor" experiment;
  - optional label-first structured rationale loss (--rationales): the verdict
    loss and inference prompt stay unchanged, while validated text-grounded
    concepts provide a small auxiliary signal;
  - eval = macro F1 with per-category threshold tuning on the val fold;
  - optional MLflow tracking (--mlflow_experiment), degrades to prints.

Usage (DataSphere H100):
  python train_lora.py --model_path /shared_models/Qwen/Qwen3.5-4B \
      --data data/lora_dataset.jsonl --val_fold 0 --output_dir runs/exp1
  python train_lora.py ... --soft_labels          # experiment 2
  python train_lora.py ... --category БАД         # per-category experiment 4
  python train_lora.py ... --rationales data/evidence.jsonl \
      --rationale_weight 0.1                      # rationale ablation
  python train_lora.py ... --data data/policy_q0.jsonl \
      --policy_query_field policy_query           # verdict-only policy pilot
Smoke test without GPU/torch: add --dry_run.
"""

import os as _os
# Must be set before torch initialises CUDA — cuts fragmentation on small cards.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import collections
import csv
import hashlib
import json
import math
import time
from pathlib import Path


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

# Policy-query rows are an intentionally narrow, verdict-only pilot.  The
# query is not product evidence and must never be hidden inside DESCRIPTION:
# description truncation is allowed, truncating this instruction is not.
POLICY_QUERY_FIELD = "policy_query"
QPAIR_POLICY_QUERY_FIELD = "qpair_policy_query"
POLICY_QUERY_ALLOWED_FIELDS = frozenset({POLICY_QUERY_FIELD})
POLICY_QUERY_DATA_SCHEMA_VERSION = "policy-query-v2-text-only"
POLICY_QUERY_PROMPT_CONTRACT_VERSION = "policy-query-prompt-v1"
POLICY_QUERY_BLOCK_TEMPLATE = (
    "<<<POLICY_QUERY_INSTRUCTION_BEGIN>>>\n"
    "{policy_query}\n"
    "<<<POLICY_QUERY_INSTRUCTION_END>>>\n\n"
)
_POLICY_QUERY_PROMPT_CONTRACT_PAYLOAD = {
    "version": POLICY_QUERY_PROMPT_CONTRACT_VERSION,
    "allowed_fields": sorted(POLICY_QUERY_ALLOWED_FIELDS),
    "block_template": POLICY_QUERY_BLOCK_TEMPLATE,
    "placement": "before_original_user_template",
    "truncation": "description_only_fail_if_fixed_prompt_exceeds_budget",
    "semantics": "policy_instruction_not_product_evidence",
}
POLICY_QUERY_PROMPT_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _POLICY_QUERY_PROMPT_CONTRACT_PAYLOAD,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

# Distilling Step-by-Step (Hsieh et al. 2023) compared two layouts and chose
# this one: rationale and label are SEPARATE tasks over the same input, never
# conditioned on each other. The joint layout the previous run used
# (prompt -> verdict -> rationale) teaches post-hoc rationalisation — the model
# explains a verdict it has already emitted — and measured -0.0045 macro with a
# median YES-logit shift of -1.44 on correct rationales. Here the rationale is
# derived from the card alone, so learning it shapes the card representation,
# and inference still reads one verdict token with no generation.
RATIONALE_TEMPLATE = (
    "Категория риска: {category}\n"
    "Название: {name}\n"
    "Описание: {description}\n\n"
    "Разбери товар по схеме ТИП/ЭЛЕМЕНТ/ВИД/РОЛЬ/СОСТОЯНИЕ/МЕХАНИЗМ:"
)

TRAINING_CONFIG_SCHEMA = "text-training-config-v1"
LEGACY_SAMPLER_MODE = "legacy_shuffle"
FIXED_RUN_SAMPLER_MODES = ("fixed_run_random", "fixed_run_stratified")


def hash_training_config(config):
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_policy_query_field(field):
    """Return the normalized optional field name, rejecting broad injection.

    This is deliberately not an arbitrary JSON field CLI: the pilot has one
    audited contract and cannot silently be repurposed to prepend rationales,
    labels or counterfactual answers.
    """

    if field in (None, ""):
        return None
    field = str(field)
    if field not in POLICY_QUERY_ALLOWED_FIELDS:
        raise ValueError(
            f"policy query field must be exactly {POLICY_QUERY_FIELD!r}, got {field!r}"
        )
    return field


def validate_qpair_policy_query_field(field):
    if field in (None, ""):
        return None
    if str(field) != QPAIR_POLICY_QUERY_FIELD:
        raise ValueError(
            f"qpair field must be exactly {QPAIR_POLICY_QUERY_FIELD!r}, got {field!r}"
        )
    return str(field)


def qpair_query_value(rec, field):
    field = validate_qpair_policy_query_field(field)
    if field is None or not bool(rec.get("qpair_active", False)):
        return None
    value = rec.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"id={rec.get('id')}: active qpair requires non-empty {field}")
    if int(rec.get("qpair_source_id", -1)) != int(rec.get("id", -2)):
        raise ValueError(f"id={rec.get('id')}: qpair source identity mismatch")
    return value.strip()


def _policy_query_value(rec, field):
    field = validate_policy_query_field(field)
    if field is None:
        return None
    if field not in rec:
        raise ValueError(f"id={rec.get('id')}: missing required field {field!r}")
    value = rec[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"id={rec.get('id')}: required field {field!r} must be a non-empty string"
        )
    return value.strip()


def policy_query_data_contract(path, field):
    """Bind a policy-query run to its exact id->query contract.

    The ordinary data SHA already protects the whole manifest.  This focused
    projection is also stored explicitly so q0, qmulti and specialist views
    remain distinguishable in experiment tables even when filenames are
    renamed, and so reviewers can audit the instruction channel directly.
    """

    field = validate_policy_query_field(field)
    if field is None:
        return {
            "version": None,
            "field": None,
            "prompt_contract_sha256": None,
            "query_projection_sha256": None,
            "schema_versions": [],
            "rows": 0,
        }

    rows = []
    seen_ids = set()
    schema_versions = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            product_id = int(rec["id"])
            if product_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id={product_id}")
            seen_ids.add(product_id)
            query = _policy_query_value(rec, field)
            schema_version = str(rec.get("policy_query_schema_version", "")).strip()
            if schema_version != POLICY_QUERY_DATA_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: policy_query_schema_version must be "
                    f"{POLICY_QUERY_DATA_SCHEMA_VERSION!r}, got {schema_version!r}; "
                    "rebuild stale policy-query manifests"
                )
            schema_versions.add(schema_version)
            rows.append(
                {
                    "id": product_id,
                    "schema_version": schema_version,
                    "query_id": str(rec.get("policy_query_id", "")),
                    "query": query,
                }
            )
    if not rows:
        raise ValueError(f"{path}: no policy-query rows")
    rows.sort(key=lambda row: row["id"])
    projection_sha256 = hashlib.sha256(
        json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": POLICY_QUERY_PROMPT_CONTRACT_VERSION,
        "field": field,
        "prompt_contract_sha256": POLICY_QUERY_PROMPT_CONTRACT_SHA256,
        "query_projection_sha256": projection_sha256,
        "schema_versions": sorted(schema_versions),
        "rows": len(rows),
    }


def resolve_policy_query_field(explicit_field, checkpoint_args=None,
                               training_config=None):
    """Resolve evaluation field from checkpoint metadata and an optional CLI.

    An explicit flag may fill missing legacy metadata, but it may never override
    a different stored contract.
    """

    explicit = validate_policy_query_field(explicit_field)
    stored = []
    for source in (checkpoint_args or {}, training_config or {}):
        value = validate_policy_query_field(source.get("policy_query_field"))
        if value is not None:
            stored.append(value)
    if len(set(stored)) > 1:
        raise ValueError(f"checkpoint has conflicting policy query fields: {stored}")
    checkpoint_field = stored[0] if stored else None
    if explicit is not None and checkpoint_field is not None and explicit != checkpoint_field:
        raise ValueError(
            f"explicit policy query field {explicit!r} conflicts with checkpoint "
            f"field {checkpoint_field!r}"
        )
    return checkpoint_field or explicit


def training_config_fingerprint(args, training_data_sha256):
    """Return an outer-fold-invariant semantic training config and its hash.

    `val_fold`, output paths, logging and evaluation cadence are deliberately
    excluded: five outer adapters must differ in fold/path while sharing every
    choice that can change the fitted model. Optional rationale inputs are bound
    by content hash rather than a machine-specific filename.
    """
    def optional_file_sha(path):
        if not path:
            return ""
        file_path = Path(path)
        if not file_path.is_file():
            raise ValueError(f"training side input does not exist: {file_path}")
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    rationale_format = str(getattr(args, "rationale_format", "grounded"))
    counterfactuals = getattr(args, "counterfactuals", None)
    counterfactual_source_bank = getattr(args, "counterfactual_source_bank", None)
    counterfactual_weight = float(getattr(args, "counterfactual_weight", 0.0) or 0.0)
    policy_query_field = validate_policy_query_field(
        getattr(args, "policy_query_field", None)
    )
    qpair_policy_query_field = validate_qpair_policy_query_field(
        getattr(args, "qpair_policy_query_field", None)
    )
    if policy_query_field is None:
        policy_query_contract = policy_query_data_contract("", None)
    else:
        data_path = getattr(args, "data", None)
        if not data_path:
            raise ValueError(
                "policy-query config fingerprint requires the training --data path"
            )
        policy_query_contract = policy_query_data_contract(
            data_path, policy_query_field
        )
    policy_query_contract_sha256 = hashlib.sha256(
        json.dumps(
            policy_query_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    deterministic_generator_sha256 = ""
    if rationale_format == "deterministic_text_rules":
        generator_path = Path(__file__).with_name("text_rule_rationales.py")
        if not generator_path.is_file():
            raise ValueError(
                f"deterministic rationale generator does not exist: {generator_path}"
            )
        deterministic_generator_sha256 = hashlib.sha256(
            generator_path.read_bytes()
        ).hexdigest()

    counterfactual_generator_sha256 = ""
    counterfactual_source_validator_sha256 = ""
    if counterfactuals:
        generator_path = Path(__file__).with_name("counterfactual_verdicts.py")
        if not generator_path.is_file():
            raise ValueError(
                f"counterfactual generator does not exist: {generator_path}"
            )
        counterfactual_generator_sha256 = hashlib.sha256(
            generator_path.read_bytes()
        ).hexdigest()
        source_validator_path = Path(__file__).with_name("text_rule_rationales.py")
        if not source_validator_path.is_file():
            raise ValueError(
                f"counterfactual source validator does not exist: {source_validator_path}"
            )
        counterfactual_source_validator_sha256 = hashlib.sha256(
            source_validator_path.read_bytes()
        ).hexdigest()

    config = {
        "schema_version": TRAINING_CONFIG_SCHEMA,
        "model_path": str(Path(args.model_path).resolve()),
        "training_data_sha256": str(training_data_sha256),
        "policy_query_field": policy_query_field,
        "policy_query_contract": policy_query_contract,
        "policy_query_contract_sha256": policy_query_contract_sha256,
        "category": args.category,
        "soft_labels": bool(args.soft_labels),
        "rationales_sha256": optional_file_sha(args.rationales),
        "rationale_audit_sha256": optional_file_sha(args.rationale_audit),
        "rationale_weight": float(args.rationale_weight),
        "rationale_format": rationale_format,
        "rationale_generator_sha256": deterministic_generator_sha256,
        "rationale_preserves_prompt_budget": True,
        "rationale_static_train_padding": True,
        "rationale_stratum_reweighting": "match_verdict_category_label_mass",
        "rationale_scope": str(args.rationale_scope),
        "rationale_max_tokens": int(args.rationale_max_tokens),
        "allow_legacy_rationales": bool(args.allow_legacy_rationales),
        "allow_unaudited_rationales": bool(args.allow_unaudited_rationales),
        # The selected counterfactual JSONL is necessarily different for every
        # outer fold. Its exact hash is checkpoint provenance, not part of this
        # deliberately outer-fold-invariant semantic config hash. The same bank
        # is derivable from data + source bank + val_fold + generator code and is
        # fully rebuilt before training.
        "counterfactual_bank_derivation": (
            "deterministic_outer_fold_rebuild" if counterfactuals else ""
        ),
        "counterfactual_source_bank_sha256": optional_file_sha(
            counterfactual_source_bank
        ),
        "counterfactual_source_generator_version": str(
            getattr(args, "counterfactual_source_generator_version", "") or ""
        ),
        "counterfactual_generator_sha256": counterfactual_generator_sha256,
        "counterfactual_source_validator_sha256": (
            counterfactual_source_validator_sha256
        ),
        "counterfactual_weight": counterfactual_weight,
        "counterfactual_target": "direct_yes_verdict_token",
        "counterfactual_pair_reduction": "mean(real_no_ce,synthetic_yes_ce)",
        "counterfactual_pair_weighting": (
            "source_real_verdict_weight_equal_mass_per_policy_rule"
        ),
        "counterfactual_rule_balance": (
            "preserve_total_selected_pair_mass_no_class_or_stratum_coverage_scale"
        ),
        "counterfactual_normalizer": "all_real_verdict_mass",
        "counterfactual_cap_per_rule_outer_train": 128,
        "counterfactual_preserves_real_prompt_budget": True,
        "counterfactual_has_rationale_tokens": False,
        "rationale_mode": str(getattr(args, "rationale_mode", "joint") or "joint"),
        "class_weight_power": float(args.class_weight_power),
        "flam_ratio": (None if args.flam_ratio is None else float(args.flam_ratio)),
        "fixed_steps": (None if args.fixed_steps is None else int(args.fixed_steps)),
        "epochs": float(args.epochs),
        "lr": float(args.lr),
        "rank": int(args.rank),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "max_len": int(args.max_len),
        "seed": int(args.seed),
    }
    # Preserve every historical Qwen fingerprint byte-for-byte.  Backbone
    # provenance exists only in the explicit Gemma 4 diversity experiment.
    from backbone_compat import (
        GEMMA4_E4B_PROFILE,
        LEGACY_QWEN_PROFILE,
        BACKBONE_CONTRACT_VERSION,
        sha256_file as backbone_sha256_file,
        validate_preflight_file,
        validate_profile,
    )
    backbone_profile = validate_profile(
        str(getattr(args, "backbone_profile", LEGACY_QWEN_PROFILE))
    )
    if backbone_profile != LEGACY_QWEN_PROFILE:
        if backbone_profile != GEMMA4_E4B_PROFILE:
            raise ValueError(f"unsupported opt-in backbone: {backbone_profile}")
        preflight_path = getattr(args, "backbone_preflight", None)
        if not preflight_path:
            raise ValueError("Gemma 4 training requires --backbone_preflight")
        validate_preflight_file(
            preflight_path,
            args.model_path,
            backbone_profile,
            require_weight_hashes=True,
        )
        config.update({
            "backbone_profile": backbone_profile,
            "backbone_contract_version": BACKBONE_CONTRACT_VERSION,
            "backbone_preflight_sha256": backbone_sha256_file(preflight_path),
            "backbone_compat_code_sha256": backbone_sha256_file(
                Path(__file__).with_name("backbone_compat.py")
            ),
            "backbone_lora_scope": "model.language_model.layers.attention_and_dense_mlp",
            "backbone_verdict_tokens": ["да", "нет"],
        })
    # Keep all historical fingerprints byte-identical.  Sampler provenance is
    # present only for the isolated optimizer-window experiment; default
    # DataLoader(shuffle=True) runs receive no new keys.
    sampler_mode = str(
        getattr(args, "sampler_mode", LEGACY_SAMPLER_MODE)
        or LEGACY_SAMPLER_MODE
    )
    if sampler_mode != LEGACY_SAMPLER_MODE:
        sampler_manifest = getattr(args, "sampler_manifest", None)
        if not sampler_manifest:
            raise ValueError("fixed-run sampler requires a manifest")
        manifest_path = Path(sampler_manifest)
        if not manifest_path.is_file():
            raise ValueError(f"sampler manifest does not exist: {manifest_path}")
        from optimizer_window_sampler import SCHEMA_VERSION as SAMPLER_SCHEMA_VERSION

        config.update({
            "sampler_mode": sampler_mode,
            "sampler_contract_schema": SAMPLER_SCHEMA_VERSION,
            "sampler_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "sampler_expected_objective": "same_weighted_ce_uniform_row_marginals",
            "sampler_fixed_run_multiplicity": "matched_exact_source_multiset",
        })
    # Preserve the historical config payload/hash byte-for-byte when QPAIR is
    # disabled. The new keys exist only inside the isolated paired ablation.
    if qpair_policy_query_field is not None:
        config["qpair_policy_query_field"] = qpair_policy_query_field
        config["qpair_loss"] = "mean(primary_gold_ce,paired_gold_ce)"
        config["qpair_shared_description_truncation"] = True
    return config, hash_training_config(config)


def load_examples(path, val_fold, category=None):
    train, val = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if category and rec["category"] != category:
                continue
            if rec["fold"] == val_fold:
                val.append(rec)
            elif not int(rec.get("train_excluded", 0)):
                train.append(rec)
            # else: a label that contradicts a confident majority over its own
            # product line. Kept in validation so scoring stays honest against
            # the convention the private test carries, dropped from training so
            # the contradiction is not learned and generalised to the family.
    return train, val


def build_user_text(rec, max_desc_chars=3000, policy_query_field=None,
                    description=None):
    """Render the classifier user body.

    With no policy query this returns the historical USER_TEMPLATE byte for
    byte.  Query mode prepends one separately delimited instruction block; the
    product card itself remains unchanged.
    """

    if description is None:
        description = rec["description"][:max_desc_chars]
    body = USER_TEMPLATE.format(
        category=rec["category"],
        name=rec["name"],
        description=description,
    )
    query = _policy_query_value(rec, policy_query_field)
    if query is None:
        return body
    return POLICY_QUERY_BLOCK_TEMPLATE.format(policy_query=query) + body


def render_classifier_prompt(tokenizer, rec, description,
                             policy_query_field=None):
    """Apply the common chat template used by training and evaluation."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_text(
                rec,
                policy_query_field=policy_query_field,
                description=description,
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def encode_description_truncated(tokenizer, rec, renderer, token_budget, *,
                                 policy_query_field=None, max_desc_chars=6000,
                                 truncated_counter=None):
    """Shorten DESCRIPTION while keeping the fixed query/card prompt intact.

    Baseline mode retains the historical six-pass proportional truncation and
    pathological tail fallback.  In policy-query mode that fallback is unsafe:
    if the fixed prompt itself cannot fit, fail closed instead of dropping the
    policy instruction from the left.
    """

    token_budget = int(token_budget)
    if token_budget < 1:
        raise ValueError("prompt token budget must be positive")
    field = validate_policy_query_field(policy_query_field)
    if field is not None:
        _policy_query_value(rec, field)
    desc = str(rec["description"])[:max_desc_chars]
    ids = []
    for _ in range(6):
        ids = tokenizer(
            renderer(rec, desc), add_special_tokens=False
        )["input_ids"]
        if len(ids) <= token_budget:
            return ids
        if not desc:
            break
        if truncated_counter is not None:
            truncated_counter[0] += 1
        desc = desc[:max(0, int(len(desc) * token_budget / len(ids) * 0.9))]

    if field is not None:
        # One explicit empty-description render avoids relying on convergence of
        # the proportional heuristic.  Either the complete fixed contract fits,
        # or the run is invalid for this max_len.
        ids = tokenizer(
            renderer(rec, ""), add_special_tokens=False
        )["input_ids"]
        if len(ids) > token_budget:
            raise ValueError(
                f"id={rec.get('id')}: policy-query fixed prompt needs "
                f"{len(ids)} tokens but max_len={token_budget}; increase max_len"
            )
        return ids
    return ids[-token_budget:]


def encode_qpair_common_description(tokenizer, rec, primary_renderer,
                                    paired_renderer, token_budget,
                                    max_desc_chars=6000,
                                    truncated_counter=None):
    """Encode two query views using exactly the same description prefix.

    The longer specialist prompt must not silently receive less product text
    than q0.  We therefore shrink one shared description until *both* complete
    fixed prompts fit.  There is no left-tail fallback: if either empty-card
    prompt is too long the experiment fails closed.
    """

    token_budget = int(token_budget)
    if token_budget < 1:
        raise ValueError("prompt token budget must be positive")
    desc = str(rec["description"])[:max_desc_chars]
    for _ in range(8):
        primary = tokenizer(primary_renderer(rec, desc), add_special_tokens=False)[
            "input_ids"
        ]
        paired = tokenizer(paired_renderer(rec, desc), add_special_tokens=False)[
            "input_ids"
        ]
        longest = max(len(primary), len(paired))
        if longest <= token_budget:
            return list(primary), list(paired), desc
        if truncated_counter is not None:
            truncated_counter[0] += 1
        if not desc:
            break
        desc = desc[:max(0, int(len(desc) * token_budget / longest * 0.9))]
    primary = tokenizer(primary_renderer(rec, ""), add_special_tokens=False)["input_ids"]
    paired = tokenizer(paired_renderer(rec, ""), add_special_tokens=False)["input_ids"]
    if max(len(primary), len(paired)) > token_budget:
        raise ValueError(
            f"id={rec.get('id')}: QPAIR fixed prompt exceeds max_len={token_budget}"
        )
    return list(primary), list(paired), ""


def _binary_verdict_ce(logp_yes, logp_no, target_p):
    """Binary-token CE usable with either floats or torch tensors."""

    return -(target_p * logp_yes + (1 - target_p) * logp_no)


def _pair_average(real_negative_ce, synthetic_yes_ce):
    """Give the original NO and its synthetic YES equal mass inside a pair."""

    return 0.5 * (real_negative_ce + synthetic_yes_ce)


def normalize_counterfactual_pair_losses(
    pair_losses, pair_weights, *, real_batch_size, real_weight_scale
):
    """Normalise sparse pairs against the frozen all-real verdict mass.

    There is intentionally no inverse-coverage or category/label stratum scale.
    With uniformly sampled real rows this is an unbiased micro-batch estimate
    of ``sum(selected source weight * pair CE) / total real verdict weight``.
    """

    if real_batch_size < 1:
        raise ValueError("real_batch_size must be positive")
    if not math.isfinite(float(real_weight_scale)) or real_weight_scale <= 0:
        raise ValueError("real_weight_scale must be finite and positive")
    if len(pair_losses) != len(pair_weights):
        raise ValueError("pair losses and weights must have equal lengths")
    return sum(loss * weight for loss, weight in zip(pair_losses, pair_weights)) / (
        real_batch_size * real_weight_scale
    )


def _assemble_counterfactual_prompt(prompt_ids, max_len):
    """Return an unchanged prompt: no verdict token and no rationale is appended."""

    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("counterfactual prompt must contain at least one token")
    if len(prompt_ids) > max_len:
        raise ValueError("counterfactual prompt exceeds the classifier prompt cap")
    return prompt_ids, len(prompt_ids) - 1


def compute_counterfactual_pair_stats(examples, source_ids, class_weights):
    """Report raw pair coverage/mass without constructing a corrective scale."""

    source_ids = {int(product_id) for product_id in source_ids}
    total_mass = 0.0
    pair_mass = 0.0
    pair_rows = 0
    for rec in examples:
        key = (rec["category"], int(rec["label"]))
        weight = float(rec["weight"]) * float(class_weights[key])
        total_mass += weight
        if int(rec["id"]) in source_ids:
            if int(rec["label"]) != 0:
                raise ValueError("counterfactual source must have a real NO label")
            pair_rows += 1
            pair_mass += weight
    if total_mass <= 0:
        raise ValueError("all-real verdict mass must be positive")
    return {
        "real_rows": len(examples),
        "pair_rows": pair_rows,
        "row_coverage": pair_rows / len(examples),
        "all_real_verdict_mass": total_mass,
        "raw_pair_source_mass": pair_mass,
        "mass_coverage": pair_mass / total_mass,
        "inverse_coverage_scale": None,
        "stratum_scale": None,
    }


def compute_counterfactual_rule_scales(examples, pairs, class_weights):
    """Equalise selected-source mass across policy rules, preserving total mass.

    This is rule balancing, not sparse-coverage correction: no category/label
    stratum is expanded to the all-real mass.  If raw selected source masses are
    ``M_r``, each rule receives target mass ``sum(M_r) / R`` and scale
    ``target / M_r``. The sum of all balanced pair weights is unchanged.
    """

    examples_by_id = {int(rec["id"]): rec for rec in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("training examples require unique ids for rule balancing")
    raw_mass = {}
    rows = {}
    for source_id, pair in pairs.items():
        source_id = int(source_id)
        rec = examples_by_id.get(source_id)
        if rec is None:
            raise ValueError(f"counterfactual source id={source_id} is not outer-train")
        if int(rec["label"]) != 0:
            raise ValueError("counterfactual rule balancing requires real NO sources")
        rule_id = str(pair["source_rule_id"])
        weight = float(rec["weight"]) * float(
            class_weights[(rec["category"], int(rec["label"]))]
        )
        raw_mass[rule_id] = raw_mass.get(rule_id, 0.0) + weight
        rows[rule_id] = rows.get(rule_id, 0) + 1
    if not raw_mass or any(mass <= 0 for mass in raw_mass.values()):
        raise ValueError("every selected counterfactual rule needs positive source mass")
    total_mass = sum(raw_mass.values())
    target_mass = total_mass / len(raw_mass)
    scales = {rule_id: target_mass / mass for rule_id, mass in raw_mass.items()}
    stats = {
        rule_id: {
            "rows": rows[rule_id],
            "raw_source_mass": raw_mass[rule_id],
            "scale": scales[rule_id],
            "balanced_source_mass": raw_mass[rule_id] * scales[rule_id],
        }
        for rule_id in sorted(raw_mass)
    }
    if not math.isclose(
        sum(item["balanced_source_mass"] for item in stats.values()),
        total_mass,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("rule balancing changed total selected pair mass")
    return scales, {
        "policy": "equal_selected_source_mass_per_rule",
        "rules": stats,
        "raw_total_pair_mass": total_mass,
        "balanced_total_pair_mass": total_mass,
        "target_mass_per_rule": target_mass,
        "class_or_stratum_coverage_scale": None,
    }


def load_counterfactual_pairs(
    path,
    source_bank_path,
    *,
    source_generator_version,
    train_records,
    validation_records,
    val_fold,
):
    """Load only an exact outer-fold rebuild of the deterministic pair bank."""

    from counterfactual_verdicts import sha256_file, validate_bank

    def read_rows(jsonl_path):
        rows = []
        with open(jsonl_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{jsonl_path}:{line_number}: JSONL row must be an object"
                    )
                rows.append(row)
        if not rows:
            raise ValueError(f"{jsonl_path}: no rows")
        return rows

    source_bank_sha256 = sha256_file(source_bank_path)
    actual = read_rows(path)
    source_rules = read_rows(source_bank_path)
    rebuilt, stats = validate_bank(
        actual,
        list(train_records) + list(validation_records),
        source_rules,
        outer_val_fold=val_fold,
        source_bank_sha256=source_bank_sha256,
        source_generator_version=source_generator_version,
    )
    pairs = {}
    for record in rebuilt:
        source_id = int(record["source_id"])
        if source_id in pairs:
            raise ValueError(f"duplicate selected counterfactual source id={source_id}")
        pairs[source_id] = record
    if not pairs:
        raise ValueError("no deterministic counterfactual pairs match outer training")
    return pairs, stats


def _reserve_label_first_budget(max_len, aux_ids):
    """Return the unchanged prompt budget and complete auxiliary ids.

    ``max_len`` is the classifier prompt cap used by the frozen baseline.  The
    verdict and rationale are appended *after* that cap for auxiliary rows.
    Causal masking means later rationale tokens cannot change the verdict
    logit, and preserving all prompt tokens avoids a hidden shorter-description
    confound in the rationale ablation.
    """
    if max_len < 1:
        raise ValueError("max_len must be positive")
    aux_ids = list(aux_ids)
    return max_len, aux_ids


def _encode_complete_rationale(tokenizer, target, max_tokens, product_id=None,
                               require_rule=True):
    """Encode one rationale without silently truncating its structural tail.

    This helper deliberately relies only on the tiny tokenizer ``encode`` API,
    so its fail-closed behaviour is unit-testable without importing
    transformers or downloading a model tokenizer.
    """

    lines = [line.rstrip() for line in str(target or "").splitlines()
             if line.strip()]
    if not lines:
        raise ValueError(f"empty rationale target for id={product_id}")

    # New structured targets must have exactly one terminal policy line. Keep
    # legacy free-form rationales usable behind their explicit CLI opt-in.
    structured = any(
        line.startswith(("ТИП:", "ЭЛЕМЕНТ:", "ПРИЧИНА:", "КОНТРАСТ:"))
        for line in lines
    )
    rule_positions = [
        index for index, line in enumerate(lines) if line.startswith("ПРАВИЛО:")
    ]
    # vlm_slots targets describe the product structure only. The policy rule id
    # is deliberately absent: measured on all 3912 FLAM cards the positive rules
    # fire at 5-20% precision against gold, so writing one into the target would
    # teach the student a conclusion we know to be wrong. The verdict head still
    # learns what to do with the structure, from the labels.
    if require_rule and structured and rule_positions != [len(lines) - 1]:
        raise ValueError(
            "structured rationale must end with exactly one ПРАВИЛО line "
            f"for id={product_id}"
        )

    ids = tokenizer.encode(
        "\n" + "\n".join(lines), add_special_tokens=False
    )
    if len(ids) > int(max_tokens):
        raise ValueError(
            f"complete rationale for id={product_id} needs {len(ids)} tokens, "
            f"exceeding --rationale_max_tokens={int(max_tokens)}; increase the "
            "limit instead of training on a truncated structured target"
        )
    return list(ids)


def _assemble_label_first_sequence(prompt_ids, verdict_id, aux_ids, max_len):
    """Assemble one auxiliary sequence and return its verdict-logit index.

    For a prompt of length ``P``, logit ``P-1`` predicts the verdict.  After the
    verdict is appended, logits ``P .. P+A-1`` predict the ``A`` rationale
    tokens.  These identities are asserted here so later refactors cannot
    silently move either loss to the wrong causal positions.
    """
    prompt_ids, aux_ids = list(prompt_ids), list(aux_ids)
    if not prompt_ids:
        raise ValueError("a label-first sequence needs at least one prompt token")
    if not aux_ids:
        raise ValueError("label-first assembly requires a non-empty rationale")

    answer_position = len(prompt_ids) - 1
    sequence = prompt_ids + [int(verdict_id)] + aux_ids
    aux_logit_start = answer_position + 1
    aux_logit_stop = aux_logit_start + len(aux_ids)

    assert len(sequence) <= max_len
    assert sequence[answer_position + 1] == verdict_id
    assert sequence[aux_logit_start + 1:aux_logit_stop + 1] == aux_ids
    assert aux_logit_stop <= len(sequence) - 1
    return sequence, answer_position


def _evidence_sources(rec):
    """Collect source enums without depending on one exact schema revision."""
    sources = set()
    for evidence in rec.get("evidence", []):
        if isinstance(evidence, dict) and evidence.get("source"):
            sources.add(str(evidence["source"]).lower())
    for element in rec.get("elements", []):
        if not isinstance(element, dict):
            continue
        for evidence in element.get("evidence", element.get("evidence_refs", [])):
            if isinstance(evidence, dict) and evidence.get("source"):
                sources.add(str(evidence["source"]).lower())
    return sources


def load_human_audit(path):
    """Return accepted ``id -> blind hash`` bindings and status counts."""
    allowed = {
        "accept", "wrong_evidence", "wrong_concept", "wrong_policy",
        "gold_conflict",
    }
    rows = []
    if Path(path).suffix.lower() == ".csv":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

    statuses, counts = {}, {status: 0 for status in sorted(allowed)}
    for row in rows:
        product_id = int(row["id"])
        status = str(row.get("human_status", "")).strip().lower()
        blind_hash = str(row.get("blind_record_sha256", "")).strip().lower()
        if status not in allowed:
            raise ValueError(
                f"audit id={product_id} has invalid human_status={status!r}")
        if product_id in statuses:
            raise ValueError(f"duplicate audit id={product_id}")
        if (len(blind_hash) != 64 or
                any(char not in "0123456789abcdef" for char in blind_hash)):
            raise ValueError(
                f"audit id={product_id} requires a valid blind_record_sha256")
        statuses[product_id] = (status, blind_hash)
        counts[status] += 1
    return {product_id: blind_hash
            for product_id, (status, blind_hash) in statuses.items()
            if status == "accept"}, counts


def load_rationale_targets(path, scope="text", allow_legacy=False,
                           source_records=None, accepted_hashes=None,
                           rationale_format="grounded"):
    """Load only validated rationale targets safe for the chosen student view.

    The text classifier never sees images at inference.  Consequently the
    default `scope=text` rejects any target whose evidence uses image/OCR facts;
    otherwise auxiliary training would explicitly teach it to hallucinate those
    facts from a product description.  `multimodal` is an opt-in ablation.

    New records must have label_support=supported.  Legacy free-text rationales
    have no grounding contract and are rejected unless explicitly allowed.

    ``deterministic_text_rules`` is a separate, deliberately small contract. Its
    completion and evidence are rebuilt from the original card by versioned
    code, so this mode does not need thousands of ceremonial human ``accept``
    rows.  It remains fail-closed: unsupported labels, source mismatches and any
    edited serialized target are rejected.
    """
    if rationale_format not in {"grounded", "deterministic_text_rules", "vlm_slots"}:
        raise ValueError(f"unknown rationale_format={rationale_format!r}")
    if rationale_format == "vlm_slots":
        # Slots extracted by the offline VLM teacher, already filtered per
        # ELEMENT to those with a name/description span, so every target is
        # derivable from what the student actually sees. The heavy grounded
        # audit does not apply: there is no human accept bank and no
        # blind_record hash — the guarantee is the per-element text filter,
        # applied by make_vlm_slot_targets.py before this file was written.
        if scope != "text":
            raise ValueError("vlm_slots targets are text-filtered; scope must be text")
        targets, stats = {}, collections.Counter()
        # The call site passes a dict {id: record}; a plain sequence also works.
        if not source_records:
            known = None
        elif isinstance(source_records, dict):
            known = {int(k) for k in source_records}
        else:
            known = {int(r["id"]) for r in source_records}
        for line in open(path):
            rec = json.loads(line)
            pid = int(rec["id"])
            if known is not None and pid not in known:
                stats["не в обучающей выборке"] += 1
                continue
            body = "\n".join(l.rstrip() for l in str(rec.get("target") or "").splitlines()
                              if l.strip())
            if not body:
                stats["пустой"] += 1
                continue
            if pid in targets:
                raise ValueError(f"duplicate vlm_slots rationale id={pid}")
            targets[pid] = body
            stats["kept"] += 1
        return targets, dict(stats)
    deterministic = rationale_format == "deterministic_text_rules"
    if deterministic:
        if allow_legacy:
            raise ValueError("deterministic text rules cannot enable legacy rationales")
        if accepted_hashes is not None:
            raise ValueError(
                "deterministic text rules are source-rebuilt and do not use a "
                "grounded-rationale human audit"
            )
        if scope != "text":
            raise ValueError("deterministic text rules require rationale_scope=text")
        from text_rule_rationales import validate_record as validate_rule_record
    else:
        from rationale_schema import (SchemaValidationError, attach_observed_label,
                                      distillation_target)

    targets, stats = {}, {
        "read": 0, "kept": 0, "not_supported": 0,
        "visual_for_text": 0, "ungrounded": 0, "invalid_schema": 0,
        "source_mismatch": 0, "not_human_accepted": 0, "empty": 0,
        "deterministic_rebuilt": 0,
    }
    blind_keys = {
        "schema_version", "product_id", "category", "product_form", "elements",
        "modality_relation", "conflicts", "counterfactual", "conclusion",
        "official_policy_verdict", "policy_rule_ids",
    }
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            stats["read"] += 1
            try:
                audit_product_id = int(rec["id"])
            except (KeyError, TypeError, ValueError):
                stats["invalid_schema"] += 1
                continue

            if deterministic:
                source = (source_records or {}).get(audit_product_id)
                if source is None:
                    stats["source_mismatch"] += 1
                    continue
                try:
                    verified = validate_rule_record(rec, source)
                    product_id = int(verified["id"])
                    if str(verified.get("student_scope", "")).lower() != "text":
                        raise ValueError("deterministic rationale is not text scoped")
                    if str(verified.get("label_support", "")).lower() != "supported":
                        stats["not_supported"] += 1
                        continue
                    target = verified["target"]
                    stats["deterministic_rebuilt"] += 1
                except (KeyError, TypeError, ValueError):
                    stats["invalid_schema"] += 1
                    continue
            else:
                if accepted_hashes is not None:
                    audited_hash = accepted_hashes.get(audit_product_id)
                    stored_hash = str(rec.get("blind_record_sha256", "")).lower()
                    if audited_hash is None or stored_hash != audited_hash:
                        stats["not_human_accepted"] += 1
                        continue
                support = str(rec.get("label_support", "")).lower()
                legacy = not support
                if legacy and not allow_legacy:
                    stats["ungrounded"] += 1
                    continue

                if legacy:
                    target = rec.get("rationale") or rec.get("structured_rationale")
                    product_id = int(rec["id"])
                else:
                    if support != "supported":
                        stats["not_supported"] += 1
                        continue
                    try:
                        product_id = int(rec["id"])
                        if str(rec.get("product_id")) != str(product_id):
                            raise ValueError("id does not match product_id")
                        observed_label = int(rec["observed_label"])
                        if observed_label not in {0, 1}:
                            raise ValueError("observed_label must be 0/1")

                        source = (source_records or {}).get(product_id)
                        if source_records is not None and source is None:
                            stats["source_mismatch"] += 1
                            continue
                        if source is not None:
                            if rec.get("category") != source.get("category") or \
                                    observed_label != int(source.get("label")):
                                stats["source_mismatch"] += 1
                                continue
                            context = {
                                "name_text": source.get("name"),
                                "description_text": source.get("description"),
                                "ocr_text_by_image": source.get("ocr_text_by_image"),
                                "known_image_ids": source.get("images", []),
                            }
                        else:
                            context = {}

                        blind = {key: value for key, value in rec.items()
                                 if key in blind_keys}
                        verified = attach_observed_label(
                            blind, observed_label, **context)
                        if (verified["label_support"] != rec.get("label_support") or
                                verified["blind_record_sha256"] !=
                                rec.get("blind_record_sha256")):
                            raise ValueError("stored label_support/hash failed recomputation")

                        sources = _evidence_sources(verified)
                        student_scope = str(rec.get("student_scope", "")).lower()
                        if scope == "text":
                            visual = any(s in {"image", "image_visual", "image_ocr", "ocr"}
                                         for s in sources)
                            if visual or student_scope in {"visual", "multimodal"}:
                                stats["visual_for_text"] += 1
                                continue
                            if not sources or student_scope != "text":
                                stats["ungrounded"] += 1
                                continue
                        # Never trust a serialized completion: rebuild it from the
                        # revalidated concepts so label-conditioned payload cannot
                        # enter training through `student_target`.
                        target = distillation_target(verified)
                    except (SchemaValidationError, KeyError, TypeError, ValueError):
                        stats["invalid_schema"] += 1
                        continue

            target = "\n".join(line.rstrip() for line in str(target or "").splitlines()
                               if line.strip())
            if not target:
                stats["empty"] += 1
                continue
            if product_id in targets:
                raise ValueError(f"duplicate eligible rationale id={product_id} in {path}")
            targets[product_id] = target
            stats["kept"] += 1
    return targets, stats


def compute_class_weights(examples, power=1.0, ratios=None):
    """Per-category weights parameterised by the positive:negative RATIO.

    The knob that matters is r = w1/w0, not the absolute weights.  Every row
    already has a non-uniform base weight (duplicate/hard-example importance),
    so class-weight normalisation must use its *weighted mass*, not raw counts.
    For one category let S0/S1 be the sums of those base weights. Then:

        Z = (S0 + r*S1) / (S0 + S1),   w0 = 1/Z,   w1 = r/Z

    Hence sum(base_weight * class_weight) remains exactly S0+S1 for every r.
    Changing the positive:negative ratio therefore cannot silently change how
    much the whole category counts against the other category.

    r defaults to (S0 / S1) ** power.  Thus power=0 adds no class weighting and
    power=1 makes the two classes contribute equal *effective* loss mass after
    the existing row weights. Equal mass is a baseline choice, not an F1-optimal
    theorem. `ratios` overrides r per category so a sweep can vary one class
    only (for example, --flam_ratio 3 means exactly w_pos/w_neg=3).
    """
    mass = {}
    for example in examples:
        key = (example["category"], int(example["label"]))
        base_weight = float(example.get("weight", 1.0))
        if not math.isfinite(base_weight) or base_weight <= 0:
            raise ValueError(
                f"base example weight must be finite and positive, got "
                f"{base_weight!r} for id={example.get('id')}")
        mass[key] = mass.get(key, 0.0) + base_weight

    weights = {}
    for cat in {c for c, _ in mass}:
        s0, s1 = mass.get((cat, 0), 0.0), mass.get((cat, 1), 0.0)
        if s0 <= 0 or s1 <= 0:
            raise ValueError(
                f"category {cat!r} needs positive base mass in both classes; "
                f"got S0={s0}, S1={s1}")
        ratio = (ratios or {}).get(cat, (s0 / s1) ** power)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                f"class ratio must be finite and positive for {cat!r}, got {ratio!r}")
        z = (s0 + ratio * s1) / (s0 + s1)
        weights[(cat, 0)] = 1.0 / z
        weights[(cat, 1)] = ratio / z
    return weights


def compute_rationale_stratum_scales(examples, rationale_ids, class_weights):
    """Correct auxiliary coverage imbalance without changing verdict weights.

    Conservative rules explain some category/label strata more often than
    others (notably many device negatives and few unambiguous FLAM positives).
    If raw auxiliary weights were used, that availability pattern would become
    a second, accidental class-weight policy.  Scale each covered stratum so
    its total auxiliary mass matches the same stratum's verdict-loss mass.

    This does not pretend the explained examples are content-random; it merely
    prevents target *coverage* from silently changing category/label priors.
    """
    rationale_ids = {int(product_id) for product_id in rationale_ids}
    total_mass, auxiliary_mass = {}, {}
    counts, auxiliary_counts = {}, {}
    for rec in examples:
        key = (rec["category"], int(rec["label"]))
        weight = float(rec["weight"]) * float(class_weights[key])
        total_mass[key] = total_mass.get(key, 0.0) + weight
        counts[key] = counts.get(key, 0) + 1
        if int(rec["id"]) in rationale_ids:
            auxiliary_mass[key] = auxiliary_mass.get(key, 0.0) + weight
            auxiliary_counts[key] = auxiliary_counts.get(key, 0) + 1

    scales = {}
    stats = {}
    for key in sorted(total_mass):
        available = auxiliary_mass.get(key, 0.0)
        scales[key] = total_mass[key] / available if available > 0 else 0.0
        stats[key] = {
            "rows": counts[key],
            "rationale_rows": auxiliary_counts.get(key, 0),
            "coverage": auxiliary_counts.get(key, 0) / counts[key],
            "verdict_mass": total_mass[key],
            "raw_rationale_mass": available,
            "scale": scales[key],
        }
    return scales, stats


def _best_threshold(y, p):
    """Threshold maximising F1 on exactly the data given."""
    import numpy as np
    from sklearn.metrics import f1_score

    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def tune_threshold_macro_f1(y_true, p_yes, categories, seed=0, n_splits=5):
    """Per-category thresholds plus an honest and an optimistic macro F1.

    Picking the threshold on the same rows you then score is peeking, so the
    reported score cross-fits: the threshold applied to each part is fitted on
    the others. n_splits controls how much data that fit gets — with 5 folds it
    sees 80%, which matters enormously for Легковоспламеняющиеся, where a val
    fold holds only ~40 positives and a 50/50 split would fit the threshold on
    ~20 of them. Too small a fitting sample makes the "honest" number
    pessimistic rather than merely unbiased.

    The thresholds handed back for deployment are still fitted on everything —
    only the *score* was ever biased, never the threshold itself.

    Returns (honest_macro_f1, optimistic_macro_f1, thresholds, per_category)
    where per_category maps category -> (honest_f1, optimistic_f1, n_positives).
    """
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    y_true, p_yes = np.asarray(y_true), np.asarray(p_yes)
    categories = np.asarray(categories)
    honest_f1s, optimistic_f1s, thresholds, per_cat = [], [], {}, {}

    for cat in np.unique(categories):
        m = categories == cat
        y_cat, p_cat = y_true[m], p_yes[m]
        n_pos = int((y_cat == 1).sum())

        t_all, f1_all = _best_threshold(y_cat, p_cat)
        thresholds[str(cat)] = round(t_all, 3)
        optimistic_f1s.append(f1_all)

        # Cross-fitting needs at least one of each class in every fold.
        k = min(n_splits, n_pos, int((y_cat == 0).sum()))
        if k < 2:
            honest = f1_all
        else:
            preds = np.zeros_like(y_cat)
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            for fit_idx, score_idx in skf.split(p_cat.reshape(-1, 1), y_cat):
                t, _ = _best_threshold(y_cat[fit_idx], p_cat[fit_idx])
                preds[score_idx] = (p_cat[score_idx] >= t).astype(int)
            honest = f1_score(y_cat, preds, zero_division=0)

        honest_f1s.append(honest)
        per_cat[str(cat)] = (round(float(honest), 4), round(float(f1_all), 4), n_pos)

    return (float(np.mean(honest_f1s)), float(np.mean(optimistic_f1s)),
            thresholds, per_cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/shared_models/Qwen/Qwen3.5-4B")
    from backbone_compat import (
        BACKBONE_PROFILES,
        GEMMA4_E4B_PROFILE,
        LEGACY_QWEN_PROFILE,
    )
    ap.add_argument(
        "--backbone_profile",
        choices=BACKBONE_PROFILES,
        default=LEGACY_QWEN_PROFILE,
        help=("explicit loader/LoRA contract; the default preserves the historical "
              "Qwen3.5-4B path byte-for-byte"),
    )
    ap.add_argument(
        "--backbone_preflight",
        default=None,
        help="hash-bound local preflight required only by gemma4_e4b_text_lora_v1",
    )
    ap.add_argument("--data", default="data/lora_dataset.jsonl")
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--output_dir", default="runs/exp")
    ap.add_argument("--category", default=None, help="train on one category only")
    ap.add_argument("--soft_labels", action="store_true")
    ap.add_argument(
        "--policy_query_field",
        choices=(POLICY_QUERY_FIELD,),
        default=None,
        help=("prepend the audited policy_query instruction from every row; "
              "verdict-only pilot, incompatible with rationale/counterfactual modes"),
    )
    ap.add_argument(
        "--qpair_policy_query_field",
        choices=(QPAIR_POLICY_QUERY_FIELD,),
        default=None,
        help=("QPAIR-DUAL paired gold-only view; requires --policy_query_field "
              "policy_query and audited matched QPAIR data"),
    )
    ap.add_argument("--rationales", default=None,
                    help="JSONL with validated structured rationales, joined by id")
    ap.add_argument("--rationale_audit", default=None,
                    help="CSV/JSONL with id,human_status; only status=accept is trained")
    ap.add_argument("--rationale_weight", type=float, default=0.0,
                    help="lambda for label-first auxiliary LM loss; first sweep: 0/.05/.1/.2")
    ap.add_argument(
        "--rationale_format",
        choices=("grounded", "deterministic_text_rules", "vlm_slots"),
        default="grounded",
        help=("grounded = audited VLM schema; deterministic_text_rules = "
              "source-rebuilt three-line text targets; vlm_slots = offline VLM "
              "teacher slots, filtered per element to text-grounded ones"),
    )
    ap.add_argument("--rationale_mode", choices=("joint", "multitask", "prompt"),
                    default="joint",
                    help="joint: prompt->verdict->rationale (прежнее поведение). "
                         "multitask: рационале — отдельная задача от входа (DSS). "
                         "prompt: слоты ДОПИСЫВАЮТСЯ ВО ВХОД перед вопросом — "
                         "оракульный замер потолка, требует слотов и на инференсе")
    ap.add_argument("--rationale_scope", choices=("text", "multimodal"), default="text",
                    help="text rejects image-grounded targets because images are absent at inference")
    ap.add_argument("--rationale_max_tokens", type=int, default=96)
    ap.add_argument("--allow_legacy_rationales", action="store_true",
                    help="allow old ungrounded free-text targets (diagnostic ablation only)")
    ap.add_argument("--allow_unaudited_rationales", action="store_true",
                    help="unsafe diagnostic ablation: train generated targets without human accept")
    ap.add_argument(
        "--counterfactuals",
        default=None,
        help="outer-fold deterministic paired counterfactual verdict JSONL",
    )
    ap.add_argument(
        "--counterfactual_source_bank",
        default=None,
        help="exact deterministic text-rule bank used to rebuild counterfactuals",
    )
    ap.add_argument(
        "--counterfactual_source_generator_version",
        default=None,
        help="explicit generator_version pinned by the source text-rule bank",
    )
    ap.add_argument(
        "--counterfactual_weight",
        type=float,
        default=None,
        help="lambda for direct paired verdict CE (default: 0.05 when mode is enabled)",
    )
    ap.add_argument("--class_weight_power", type=float, default=1.0,
                    help="r = (S_neg/S_pos)**power по исходным row weights; "
                         "1.0 = равная effective loss mass классов, 0 = без class weights")
    ap.add_argument("--flam_pos_frac", type=float, default=1.0,
                    help="доля позитивов Легковоспламеняющихся, оставляемых в "
                         "ОБУЧЕНИИ (val не трогается). Для замера кривой "
                         "качество-против-числа-примеров редкого класса.")
    ap.add_argument("--flam_ratio", type=float, default=None,
                    help="явное отношение вес(поз)/вес(нег) для Легковоспламеняющихся; "
                         "при неизменном --class_weight_power меняет только эту категорию")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--fixed_steps", type=int, default=None,
                    help="train exactly N steps without evaluating/selecting on val; "
                         "required for untouched outer-fold predictions")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_batch_size", type=int, default=None,
                    help="defaults to batch_size*8 — eval needs no backward pass")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sampler_mode",
        choices=(LEGACY_SAMPLER_MODE, *FIXED_RUN_SAMPLER_MODES),
        default=LEGACY_SAMPLER_MODE,
        help=("isolated optimizer-window experiment; default preserves historical "
              "DataLoader shuffle byte-for-byte"),
    )
    ap.add_argument(
        "--sampler_manifest",
        default=None,
        help="frozen parity manifest required by fixed_run_* sampler modes",
    )
    ap.add_argument("--mlflow_experiment", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.backbone_profile == LEGACY_QWEN_PROFILE:
        if args.backbone_preflight:
            ap.error("--backbone_preflight is forbidden for the legacy Qwen profile")
    else:
        if args.backbone_profile != GEMMA4_E4B_PROFILE:
            ap.error(f"unsupported opt-in backbone profile: {args.backbone_profile}")
        if not args.backbone_preflight:
            ap.error("Gemma 4 requires --backbone_preflight")
        # Keep the first backbone comparison single-variable and deployable.
        # Rationale/QPAIR/counterfactual/sampler branches have their own gates.
        if args.category is not None or args.soft_labels:
            ap.error("Gemma 4 one-fold pilot requires joint hard-label training")
        if args.fixed_steps is None:
            ap.error("Gemma 4 one-fold pilot requires --fixed_steps")
        if (
            args.policy_query_field or args.qpair_policy_query_field
            or args.rationales or args.rationale_audit or args.rationale_weight != 0
            or args.rationale_mode != "joint"
            or args.counterfactuals or args.counterfactual_source_bank
            or args.counterfactual_source_generator_version
            or args.counterfactual_weight not in (None, 0.0)
            or args.sampler_mode != LEGACY_SAMPLER_MODE
        ):
            ap.error(
                "Gemma 4 diversity pilot is verdict-only: no policy-query, "
                "rationale, counterfactual or sampler intervention"
            )

    if args.sampler_mode == LEGACY_SAMPLER_MODE:
        if args.sampler_manifest:
            ap.error("--sampler_manifest is forbidden with legacy_shuffle")
    else:
        if not args.sampler_manifest:
            ap.error("fixed_run_* sampler modes require --sampler_manifest")
        if args.fixed_steps is None:
            ap.error("fixed_run_* sampler modes require --fixed_steps")
        if args.category is not None:
            ap.error("fixed-run sampler pilot requires the unchanged joint-category task")
        if args.soft_labels:
            ap.error("fixed-run sampler pilot requires unchanged hard-label weighted CE")
        if args.flam_pos_frac != 1.0:
            ap.error("fixed-run sampler pilot forbids FLAM row subsampling")
        if (
            args.policy_query_field or args.qpair_policy_query_field
            or args.rationales or args.rationale_audit or args.rationale_weight != 0
            or args.counterfactuals or args.counterfactual_source_bank
            or args.counterfactual_source_generator_version
            or args.counterfactual_weight not in (None, 0.0)
        ):
            ap.error(
                "fixed-run sampler pilot is isolated: no policy-query, rationale, "
                "QPAIR or counterfactual intervention is allowed"
            )

    if args.qpair_policy_query_field and args.policy_query_field != POLICY_QUERY_FIELD:
        ap.error("--qpair_policy_query_field requires --policy_query_field policy_query")

    if args.policy_query_field:
        if (
            args.rationales
            or args.rationale_audit
            or args.rationale_weight != 0
            or args.rationale_mode != "joint"
            or args.allow_legacy_rationales
            or args.allow_unaudited_rationales
        ):
            ap.error(
                "--policy_query_field is a verdict-only pilot and forbids all "
                "rationale inputs/modes"
            )
        if (
            args.counterfactuals
            or args.counterfactual_source_bank
            or args.counterfactual_source_generator_version
            or args.counterfactual_weight not in (None, 0.0)
        ):
            ap.error(
                "--policy_query_field forbids counterfactual verdict mode in this pilot"
            )

    if args.counterfactuals:
        if args.counterfactual_weight is None:
            args.counterfactual_weight = 0.05
        if not args.counterfactual_source_bank:
            ap.error("--counterfactuals requires --counterfactual_source_bank")
        if not args.counterfactual_source_generator_version:
            ap.error(
                "--counterfactuals requires --counterfactual_source_generator_version"
            )
        if args.rationale_weight != 0 or args.rationales or args.rationale_audit:
            ap.error(
                "counterfactual verdict mode requires rationale_weight=0 and no "
                "rationale bank/audit; it trains no rationale tokens"
            )
        if args.allow_legacy_rationales or args.allow_unaudited_rationales:
            ap.error("counterfactual verdict mode forbids rationale escape hatches")
        if args.soft_labels:
            ap.error("counterfactual verdict mode requires hard real labels")
        if args.category == "БАД":
            ap.error("counterfactual verdict sources are flammable-category rows")
        if args.fixed_steps is None:
            ap.error(
                "counterfactual outer-fold training requires --fixed_steps so the "
                "held-out fold is never used for checkpoint selection"
            )
    else:
        if args.counterfactual_source_bank or args.counterfactual_source_generator_version:
            ap.error("counterfactual source provenance requires --counterfactuals")
        if args.counterfactual_weight not in {None, 0.0}:
            ap.error("--counterfactual_weight requires --counterfactuals")
        args.counterfactual_weight = 0.0

    if (
        not math.isfinite(args.counterfactual_weight)
        or args.counterfactual_weight < 0
    ):
        ap.error("--counterfactual_weight must be finite and non-negative")
    if args.counterfactuals and args.counterfactual_weight <= 0:
        ap.error("counterfactual verdict mode requires positive --counterfactual_weight")

    if not math.isfinite(args.rationale_weight) or args.rationale_weight < 0:
        ap.error("--rationale_weight must be non-negative")
    if args.rationale_max_tokens < 1:
        ap.error("--rationale_max_tokens must be positive")
    if args.rationale_weight > 0 and not args.rationales:
        ap.error("--rationale_weight > 0 requires --rationales")
    if (args.rationale_weight > 0 and not args.rationale_audit and
            not args.allow_unaudited_rationales and
            args.rationale_format != "deterministic_text_rules"):
        ap.error("rationale training requires --rationale_audit; use "
                 "--allow_unaudited_rationales only for an explicit diagnostic ablation")
    if args.rationale_format == "deterministic_text_rules":
        if args.rationale_audit:
            ap.error("deterministic text rules are rebuilt from source and do not use --rationale_audit")
        if args.allow_legacy_rationales or args.allow_unaudited_rationales:
            ap.error("deterministic text rules cannot use legacy/unaudited escape hatches")
        if args.rationale_scope != "text":
            ap.error("deterministic text rules require --rationale_scope text")
    if args.rationale_weight > 0 and args.soft_labels:
        ap.error("rationale pilot and --soft_labels must be separate one-variable experiments")
    # --val_fold 99 with --fixed_steps: no row carries fold 99, so every row —
    # including the curated fold=-1 rows — trains and nothing is evaluated. This
    # is the "final model on all data" mode; its adapter has no honest OOF and
    # must borrow thresholds from the fold-soup of the same recipe.
    if args.val_fold not in {0, 1, 2, 3, 4}:
        if not (args.val_fold == 99 and args.fixed_steps is not None):
            ap.error("--val_fold must be one of 0,1,2,3,4 (or 99 together with "
                     "--fixed_steps to train on every row without evaluation)")
    if args.category not in {None, "БАД", "Легковоспламеняющиеся"}:
        ap.error("--category must be БАД or Легковоспламеняющиеся")
    positive_ints = {
        "--rank": args.rank, "--batch_size": args.batch_size,
        "--grad_accum": args.grad_accum, "--eval_every": args.eval_every,
        "--log_every": args.log_every, "--max_len": args.max_len,
    }
    if args.eval_batch_size is not None:
        positive_ints["--eval_batch_size"] = args.eval_batch_size
    if args.fixed_steps is not None:
        positive_ints["--fixed_steps"] = args.fixed_steps
    for name, value in positive_ints.items():
        if value < 1:
            ap.error(f"{name} must be positive")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        ap.error("--epochs must be finite and positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error("--lr must be finite and positive")
    if not math.isfinite(args.class_weight_power) or args.class_weight_power < 0:
        ap.error("--class_weight_power must be finite and non-negative")
    if (args.flam_ratio is not None and
            (not math.isfinite(args.flam_ratio) or args.flam_ratio <= 0)):
        ap.error("--flam_ratio must be finite and positive")

    train, val = load_examples(args.data, args.val_fold, args.category)
    if args.policy_query_field:
        try:
            for rec in train + val:
                _policy_query_value(rec, args.policy_query_field)
                if args.qpair_policy_query_field:
                    qpair_query_value(rec, args.qpair_policy_query_field)
        except ValueError as exc:
            ap.error(str(exc))
    data_sha256 = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    if not train or (not val and args.val_fold != 99):
        ap.error(f"empty train/validation split: train={len(train)} val={len(val)}")
    try:
        training_config, training_config_sha256 = training_config_fingerprint(
            args, data_sha256)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    ratios = ({"Легковоспламеняющиеся": args.flam_ratio}
              if args.flam_ratio is not None else None)
    class_w = compute_class_weights(train, args.class_weight_power, ratios)
    if any(not math.isfinite(weight) or weight <= 0 for weight in class_w.values()):
        ap.error(f"class weights must be finite and positive, got {class_w}")
    print(f"train={len(train)} val={len(val)} class_weights={class_w}")
    print(f"training config sha256={training_config_sha256}")

    frac = float(getattr(args, "flam_pos_frac", 1.0) or 1.0)
    if frac < 1.0:
        # Deterministic by id, so every fraction is a nested subset: the 25%
        # run trains on a subset of what the 50% run sees. Otherwise each point
        # would resample and the curve would carry selection noise on top of
        # the effect being measured.
        def _keep(rec):
            if rec["category"] != "Легковоспламеняющиеся" or int(rec["label"]) != 1:
                return True
            h = hashlib.sha256(str(rec["id"]).encode()).hexdigest()
            return (int(h[:8], 16) / 0xFFFFFFFF) < frac
        before = sum(1 for r in train
                     if r["category"] == "Легковоспламеняющиеся" and int(r["label"]) == 1)
        train = [r for r in train if _keep(r)]
        after = sum(1 for r in train
                    if r["category"] == "Легковоспламеняющиеся" and int(r["label"]) == 1)
        print(f"flam_pos_frac={frac}: позитивов FLAM в обучении {before} -> {after}",
              flush=True)

    sampler_manifest = None
    if args.sampler_mode != LEGACY_SAMPLER_MODE:
        try:
            from optimizer_window_sampler import validate_manifest_for_training

            sampler_manifest = validate_manifest_for_training(
                args.sampler_manifest,
                args.data,
                train,
                mode=args.sampler_mode,
                seed=args.seed,
                fixed_steps=args.fixed_steps,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
            )
            sampler_recipe = sampler_manifest.get("recipe", {})
            sampler_exact_recipe = {
                "rank": int(args.rank),
                "max_len": int(args.max_len),
            }
            for key, actual in sampler_exact_recipe.items():
                if sampler_recipe.get(key) != actual:
                    raise ValueError(
                        f"sampler recipe mismatch for {key}: "
                        f"{sampler_recipe.get(key)!r} != {actual!r}"
                    )
            sampler_float_recipe = {
                "lr": float(args.lr),
                "class_weight_power": float(args.class_weight_power),
                "flam_ratio": float(args.flam_ratio),
            }
            for key, actual in sampler_float_recipe.items():
                if not math.isclose(
                    float(sampler_recipe.get(key, float("nan"))), actual,
                    rel_tol=0, abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"sampler recipe mismatch for {key}: "
                        f"{sampler_recipe.get(key)!r} != {actual!r}"
                    )
            sampler_split = sampler_manifest.get("split", {})
            if int(sampler_split.get("outer_fold", -1)) != int(args.val_fold):
                raise ValueError("sampler manifest outer fold differs from --val_fold")
            if int(sampler_split.get("inner_fold", -1)) == int(args.val_fold):
                raise ValueError("sampler inner and outer folds collide")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            ap.error(f"invalid optimizer-window sampler contract: {exc}")
        print(
            "optimizer-window sampler="
            + json.dumps({
                "mode": args.sampler_mode,
                "manifest": hashlib.sha256(
                    Path(args.sampler_manifest).read_bytes()
                ).hexdigest(),
                "layout": sampler_manifest["layout"],
                "parity": sampler_manifest["parity_proof"],
            }, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    rationale_targets = {}
    if args.rationales:
        accepted_hashes = None
        if args.rationale_audit:
            try:
                accepted_hashes, audit_stats = load_human_audit(args.rationale_audit)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                ap.error(f"invalid rationale audit: {exc}")
            print(f"human audit: {audit_stats} | accepted={len(accepted_hashes)}")
        rationale_targets, rationale_stats = load_rationale_targets(
            args.rationales, scope=args.rationale_scope,
            allow_legacy=args.allow_legacy_rationales,
            source_records={int(r["id"]): r for r in
                            (train if args.fixed_steps is not None else train + val)},
            accepted_hashes=accepted_hashes,
            rationale_format=args.rationale_format)
        in_train = sum(int(r["id"]) in rationale_targets for r in train)
        print(f"rationales: {rationale_stats} | matched train={in_train} "
              f"lambda={args.rationale_weight} scope={args.rationale_scope}")
        if args.rationale_weight > 0 and in_train == 0:
            ap.error("no validated rationale targets match the training split")

    counterfactual_pairs = {}
    counterfactual_stats = {}
    counterfactual_rule_scales = {}
    counterfactual_provenance = {}
    if args.counterfactuals:
        try:
            counterfactual_pairs, counterfactual_stats = load_counterfactual_pairs(
                args.counterfactuals,
                args.counterfactual_source_bank,
                source_generator_version=(
                    args.counterfactual_source_generator_version
                ),
                train_records=train,
                validation_records=val,
                val_fold=args.val_fold,
            )
            pair_mass_stats = compute_counterfactual_pair_stats(
                train, counterfactual_pairs, class_w
            )
            (
                counterfactual_rule_scales,
                counterfactual_rule_balance_stats,
            ) = compute_counterfactual_rule_scales(
                train, counterfactual_pairs, class_w
            )
            counterfactual_provenance = {
                "outer_val_fold": args.val_fold,
                "counterfactual_bank_sha256": hashlib.sha256(
                    Path(args.counterfactuals).read_bytes()
                ).hexdigest(),
                "source_bank_sha256": hashlib.sha256(
                    Path(args.counterfactual_source_bank).read_bytes()
                ).hexdigest(),
                "source_generator_version": (
                    args.counterfactual_source_generator_version
                ),
                "selected_pairs": len(counterfactual_pairs),
                "selection_stats": counterfactual_stats,
                "pair_mass_stats": pair_mass_stats,
                "rule_balance_stats": counterfactual_rule_balance_stats,
            }
        except (OSError, KeyError, TypeError, ValueError) as exc:
            ap.error(f"invalid deterministic counterfactual bank: {exc}")
        print(
            "counterfactuals: "
            + json.dumps(counterfactual_stats, ensure_ascii=False, sort_keys=True)
        )
        print(
            "counterfactual pair mass: "
            + json.dumps(pair_mass_stats, ensure_ascii=False, sort_keys=True)
            + f" | lambda={args.counterfactual_weight}"
        )
        print(
            "counterfactual rule balance: "
            + json.dumps(
                counterfactual_rule_balance_stats,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    if args.dry_run:
        rec = train[0]
        print("--- sample prompt ---")
        print(SYSTEM_PROMPT)
        sample_user_text = build_user_text(
            rec, policy_query_field=args.policy_query_field
        )
        # The legacy dry-run stays compact.  Query mode prints the complete
        # instruction block so its exact contract can be audited before a GPU
        # job starts; only the source description was already capped by
        # build_user_text.
        print(sample_user_text if args.policy_query_field else sample_user_text[:600])
        print("--- target:", "да" if rec["label"] == 1 else "нет",
              "| soft:", rec["soft_label"], "| weight:", rec["weight"])
        if rationale_targets:
            matched = next((r for r in train if int(r["id"]) in rationale_targets), None)
            if matched:
                print("--- label-first rationale target ---")
                print(("да" if matched["label"] else "нет") + "\n" +
                      rationale_targets[int(matched["id"])])
        if counterfactual_pairs:
            from counterfactual_verdicts import counterfactual_prompt_record

            matched = next(
                r for r in train if int(r["id"]) in counterfactual_pairs
            )
            pair = counterfactual_pairs[int(matched["id"])]
            synthetic = counterfactual_prompt_record(pair)
            print("--- paired counterfactual verdict (no rationale tokens) ---")
            print(
                f"source id={matched['id']} fold={matched['fold']} "
                f"rule={pair['source_rule_id']} real target=нет synthetic target=да"
            )
            print(build_user_text(synthetic)[:600])
        print("dry run OK")
        return

    # --- heavy imports only past this point ---
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow = None
    if args.mlflow_experiment:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow.start_run()
            mlflow.log_params({k: v for k, v in vars(args).items() if v is not None})
        except ImportError:
            print("mlflow not installed — falling back to prints")

    backbone_runtime_provenance = None
    if args.backbone_profile == LEGACY_QWEN_PROFILE:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        yes_id = tokenizer.encode("да", add_special_tokens=False)
        no_id = tokenizer.encode("нет", add_special_tokens=False)
        assert len(yes_id) == 1 and len(no_id) == 1, \
            f"verdict must be single tokens, got {yes_id}/{no_id}; pick different words"
        yes_id, no_id = yes_id[0], no_id[0]

        # transformers >=5 renamed the `torch_dtype` argument to `dtype`
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, dtype=torch.bfloat16, local_files_only=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
        model = model.to(device)
        model.config.use_cache = False
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        from backbone_compat import (
            audit_peft_lora_scope,
            disable_model_cache,
            environment_versions,
            load_model_for_profile,
            load_tokenizer_for_profile,
            resolve_gemma4_lora_targets,
            sha256_file as backbone_sha256_file,
            validate_preflight_file,
            validate_tokenizer_against_preflight,
        )

        tokenizer, verdict_contract = load_tokenizer_for_profile(
            args.model_path, args.backbone_profile
        )
        gemma_preflight = validate_preflight_file(
            args.backbone_preflight,
            args.model_path,
            args.backbone_profile,
            require_weight_hashes=True,
        )
        validate_tokenizer_against_preflight(
            verdict_contract, gemma_preflight
        )
        yes_id = verdict_contract["verdicts"]["yes"]["token_id"]
        no_id = verdict_contract["verdicts"]["no"]["token_id"]
        model = load_model_for_profile(
            args.model_path, args.backbone_profile, dtype=torch.bfloat16
        ).to(device)
        disable_model_cache(model, args.backbone_profile)
        target_modules, target_audit = resolve_gemma4_lora_targets(model)
        backbone_runtime_provenance = {
            **target_audit,
            "environment": environment_versions(),
            "tokenizer": verdict_contract,
            "preflight_sha256": backbone_sha256_file(args.backbone_preflight),
        }

    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if args.backbone_profile == GEMMA4_E4B_PROFILE:
        peft_scope = audit_peft_lora_scope(model, target_modules)
        backbone_runtime_provenance["peft_scope"] = peft_scope
        print(
            "Gemma backbone audit="
            + json.dumps(backbone_runtime_provenance, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    # Order matters: with a frozen base, checkpointing only saves memory once the
    # inputs require grad — otherwise the recompute graph is never built and the
    # activations stay resident. Must come AFTER get_peft_model.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    prompt_slots = (getattr(args, "rationale_mode", "joint") == "prompt")

    def _render(rec, description):
        body = build_user_text(
            rec,
            policy_query_field=args.policy_query_field,
            description=description,
        )
        if prompt_slots:
            # Slots go BEFORE the question so the verdict logit can attend to
            # them. This is the oracle arm: it needs the same annotation at
            # inference, which only a distilled extractor could supply. Its
            # purpose is to price that extractor before anyone builds one.
            slots = rationale_targets.get(int(rec["id"]))
            if slots:
                head, _, tail = body.rpartition("\n\n")
                body = f"{head}\n\nРазбор структуры:\n{slots}\n\n{tail}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]
        # Render to text first: transformers 5 returns a tokenizers.Encoding from
        # apply_chat_template(tokenize=True), 4.x returns a plain list. Asking for
        # the string and tokenising separately behaves the same on both.
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def _render_qpair(rec, description):
        query = qpair_query_value(rec, args.qpair_policy_query_field)
        if query is None:
            raise ValueError(f"id={rec.get('id')}: inactive row has no paired render")
        paired = dict(rec)
        paired[POLICY_QUERY_FIELD] = query
        return render_classifier_prompt(
            tokenizer, paired, description,
            policy_query_field=POLICY_QUERY_FIELD,
        )

    truncated = [0]

    def _render_rationale(rec, description):
        """Same card, different task prefix — no verdict token anywhere."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RATIONALE_TEMPLATE.format(
                category=rec["category"], name=rec["name"], description=description)},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    if counterfactual_pairs:
        from counterfactual_verdicts import counterfactual_prompt_record

    def encode(rec, token_budget=None, renderer=None):
        """Fit the prompt to its token budget by shortening DESCRIPTION.

        Cutting tokens off the front instead (ids[-max_len:]) silently drops the
        product name, which sits before the description and is often the single
        most informative field ("Мангал одноразовый в комплекте с углём").
        At max_len=512 that hit 42% of the dataset.
        """
        render = renderer or _render
        token_budget = args.max_len if token_budget is None else int(token_budget)
        if not 1 <= token_budget <= args.max_len:
            raise ValueError(
                f"prompt token budget must be in [1, {args.max_len}], "
                f"got {token_budget}")
        return encode_description_truncated(
            tokenizer,
            rec,
            render,
            token_budget,
            policy_query_field=(
                args.policy_query_field if renderer is None else None
            ),
            truncated_counter=truncated,
        )

    class VerdictDataset(Dataset):
        # Tokenise once up front rather than per __getitem__: at small batch sizes
        # the chat-template + tokenizer call costs about as much as the forward
        # pass itself, which leaves the GPU idle half the time.
        def __init__(self, examples, name="", with_aux=False):
            self.examples = examples
            t = time.time()
            before = truncated[0]
            self.encoded = []
            self.prompt_lengths = []
            self.qpair_encoded = []
            self.qpair_answer_positions = []
            self.aux_targets = []
            self.counterfactual_encoded = []
            self.counterfactual_answer_positions = []
            self.counterfactual_rule_ids = []
            self.rt_encoded = []
            self.rt_answer_positions = []
            self.rt_targets = []
            multitask = (getattr(args, "rationale_mode", "joint") == "multitask"
                         and args.rationale_weight > 0)
            for rec in examples:
                aux_ids = []
                if with_aux and args.rationale_weight > 0:
                    target = rationale_targets.get(int(rec["id"]))
                    if target:
                        aux_ids = _encode_complete_rationale(
                            tokenizer,
                            target,
                            args.rationale_max_tokens,
                            product_id=rec["id"],
                            require_rule=(getattr(args, "rationale_format", "grounded")
                                          != "vlm_slots"),
                        )
                rt_ids, rt_pos, rt_tgt = [], None, []
                if multitask and with_aux and aux_ids:
                    budget = args.max_len - len(aux_ids)
                    if budget > 32:
                        rt_prompt = encode(rec, budget, renderer=_render_rationale)
                        if rt_prompt:
                            rt_ids = list(rt_prompt) + list(aux_ids)
                            rt_pos = len(rt_prompt) - 1
                            rt_tgt = list(aux_ids)
                    # In multitask mode the verdict row must stay a plain
                    # classifier row: no rationale tokens are appended to it.
                    aux_ids = []
                prompt_budget, aux_ids = _reserve_label_first_budget(
                    args.max_len, aux_ids)
                pair_query = (
                    qpair_query_value(rec, args.qpair_policy_query_field)
                    if args.qpair_policy_query_field else None
                )
                if pair_query is not None:
                    prompt_ids, pair_ids, _shared_description = (
                        encode_qpair_common_description(
                            tokenizer, rec, _render, _render_qpair,
                            prompt_budget, truncated_counter=truncated,
                        )
                    )
                    self.qpair_encoded.append(pair_ids)
                    self.qpair_answer_positions.append(len(pair_ids) - 1)
                else:
                    prompt_ids = encode(rec, prompt_budget)
                    self.qpair_encoded.append([])
                    self.qpair_answer_positions.append(None)
                if not prompt_ids:
                    raise RuntimeError(f"empty encoded prompt for id={rec['id']}")
                self.prompt_lengths.append(len(prompt_ids))
                self.aux_targets.append(aux_ids)
                self.rt_encoded.append(rt_ids)
                self.rt_answer_positions.append(rt_pos)
                self.rt_targets.append(rt_tgt)
                if aux_ids:
                    verdict_id = yes_id if rec["label"] == 1 else no_id
                    sequence, answer_position = _assemble_label_first_sequence(
                        prompt_ids, verdict_id, aux_ids,
                        args.max_len + 1 + args.rationale_max_tokens)
                    assert answer_position == len(prompt_ids) - 1
                    self.encoded.append(sequence)
                else:
                    assert len(prompt_ids) <= args.max_len
                    self.encoded.append(prompt_ids)
                counterfactual_ids = []
                counterfactual_answer_position = None
                counterfactual_rule_id = None
                if with_aux and args.counterfactual_weight > 0:
                    pair = counterfactual_pairs.get(int(rec["id"]))
                    if pair:
                        counterfactual_rule_id = str(pair["source_rule_id"])
                        synthetic = counterfactual_prompt_record(pair)
                        encoded_synthetic = encode(synthetic, args.max_len)
                        (
                            counterfactual_ids,
                            counterfactual_answer_position,
                        ) = _assemble_counterfactual_prompt(
                            encoded_synthetic, args.max_len
                        )
                self.counterfactual_encoded.append(counterfactual_ids)
                self.counterfactual_answer_positions.append(
                    counterfactual_answer_position
                )
                self.counterfactual_rule_ids.append(counterfactual_rule_id)
            n_trunc = truncated[0] - before
            n_aux = sum(bool(x) for x in self.aux_targets)
            n_counterfactual = sum(bool(x) for x in self.counterfactual_encoded)
            print(f"tokenized {len(examples)} {name} examples in {time.time() - t:.1f}s "
                  f"({n_trunc} needed description truncation to fit prompt max_len="
                  f"{args.max_len}; {n_aux} have appended rationale targets; "
                  f"{n_counterfactual} have paired direct-verdict prompts; "
                  f"{sum(bool(x) for x in self.rt_targets)} separate rationale-task "
                  f"rows)", flush=True)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            rec = self.examples[i]
            target_p = rec["soft_label"] if args.soft_labels else float(rec["label"])
            stratum = (rec["category"], rec["label"])
            w = rec["weight"] * class_w[stratum]
            aux_w = (w * rationale_stratum_scale.get(stratum, 0.0)
                     if self.aux_targets[i] else 0.0)
            counterfactual_rule_id = self.counterfactual_rule_ids[i]
            counterfactual_pair_weight = (
                w * counterfactual_rule_scales[counterfactual_rule_id]
                if counterfactual_rule_id is not None else 0.0
            )
            return {
                "input_ids": self.encoded[i], "target_p": target_p, "weight": w,
                "aux_weight": aux_w,
                "label": rec["label"], "category": rec["category"],
                "answer_position": self.prompt_lengths[i] - 1,
                "aux_targets": self.aux_targets[i],
                "counterfactual_input_ids": self.counterfactual_encoded[i],
                "counterfactual_answer_position": (
                    self.counterfactual_answer_positions[i]
                ),
                "counterfactual_source_rule_id": counterfactual_rule_id,
                "counterfactual_pair_weight": counterfactual_pair_weight,
                "rt_input_ids": self.rt_encoded[i],
                "rt_answer_position": self.rt_answer_positions[i],
                "rt_targets": self.rt_targets[i],
                "rt_weight": (w * rationale_stratum_scale.get(stratum, 0.0)
                              if self.rt_targets[i] else 0.0),
                "qpair_input_ids": self.qpair_encoded[i],
                "qpair_answer_position": self.qpair_answer_positions[i],
            }

    def collate(batch, *, static_rationale_shape=False):
        sequence_cap = args.max_len + 1 + args.rationale_max_tokens
        maxlen = (sequence_cap if static_rationale_shape
                  else max(len(b["input_ids"]) for b in batch))
        assert max(len(b["input_ids"]) for b in batch) <= maxlen <= sequence_cap
        pad = tokenizer.pad_token_id or 0
        input_ids, attn, answer_positions = [], [], []
        for b in batch:  # left-pad so the answer position is always last
            n = maxlen - len(b["input_ids"])
            input_ids.append([pad] * n + b["input_ids"])
            attn.append([0] * n + [1] * len(b["input_ids"]))
            answer_positions.append(n + b["answer_position"])
            assert 0 <= answer_positions[-1] < maxlen
            if b["aux_targets"]:
                # The final selected auxiliary logit is immediately before the
                # final target token; it must remain inside this padded row.
                aux_stop = answer_positions[-1] + 1 + len(b["aux_targets"])
                assert aux_stop <= maxlen - 1
        result = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "answer_positions": torch.tensor(answer_positions),
            "aux_targets": [b["aux_targets"] for b in batch],
            "target_p": torch.tensor([b["target_p"] for b in batch]),
            "weight": torch.tensor([b["weight"] for b in batch]),
            "aux_weight": torch.tensor([b["aux_weight"] for b in batch]),
            "labels_int": torch.tensor([b["label"] for b in batch]),
            "categories": [b["category"] for b in batch],
        }
        qpair_rows = [
            (source_index, b) for source_index, b in enumerate(batch)
            if b.get("qpair_input_ids")
        ]
        if qpair_rows:
            qpair_maxlen = args.max_len
            qp_ids, qp_attn, qp_pos, qp_source = [], [], [], []
            for source_index, row_data in qpair_rows:
                ids = row_data["qpair_input_ids"]
                n = qpair_maxlen - len(ids)
                if n < 0:
                    raise AssertionError("paired prompt exceeds max_len")
                qp_ids.append([pad] * n + ids)
                qp_attn.append([0] * n + [1] * len(ids))
                qp_pos.append(n + int(row_data["qpair_answer_position"]))
                qp_source.append(source_index)
            result.update({
                "qpair_input_ids": torch.tensor(qp_ids),
                "qpair_attention_mask": torch.tensor(qp_attn),
                "qpair_answer_positions": torch.tensor(qp_pos),
                "qpair_source_indices": torch.tensor(qp_source),
            })
        else:
            result.update({
                "qpair_input_ids": None,
                "qpair_attention_mask": None,
                "qpair_answer_positions": None,
                "qpair_source_indices": None,
            })
        # Multitask rationale rows: same cards, different task prefix, no verdict
        # token. Kept as their own sub-batch so the verdict rows above stay
        # byte-identical to the baseline recipe and the comparison stays clean.
        rt_rows = [b for b in batch if b.get("rt_targets")]
        if rt_rows:
            rt_maxlen = max(len(b["rt_input_ids"]) for b in rt_rows)
            rt_ids, rt_attn, rt_pos, rt_tg, rt_w = [], [], [], [], []
            for b in rt_rows:
                ids = b["rt_input_ids"]
                n = rt_maxlen - len(ids)
                rt_ids.append([pad] * n + ids)
                rt_attn.append([0] * n + [1] * len(ids))
                position = n + int(b["rt_answer_position"])
                assert 0 <= position < rt_maxlen
                assert position + len(b["rt_targets"]) <= rt_maxlen - 1
                rt_pos.append(position)
                rt_tg.append(b["rt_targets"])
                rt_w.append(float(b["rt_weight"]))
            result["rt_input_ids"] = torch.tensor(rt_ids)
            result["rt_attention_mask"] = torch.tensor(rt_attn)
            result["rt_answer_positions"] = torch.tensor(rt_pos)
            result["rt_targets"] = rt_tg
            result["rt_weights"] = torch.tensor(rt_w)
        else:
            result["rt_input_ids"] = None
            result["rt_targets"] = []

        cf_rows = [
            (source_index, b)
            for source_index, b in enumerate(batch)
            if b["counterfactual_input_ids"]
        ]
        if cf_rows:
            # A single sequence length avoids a new FLA/Triton specialisation
            # for every synthetic prompt length. The real baseline rows above
            # retain their exact original prompt lengths and budget.
            cf_maxlen = args.max_len
            cf_input_ids, cf_attention, cf_positions, cf_source_indices = [], [], [], []
            cf_pair_weights, cf_rule_ids = [], []
            for source_index, row_data in cf_rows:
                ids = row_data["counterfactual_input_ids"]
                n = cf_maxlen - len(ids)
                cf_input_ids.append([pad] * n + ids)
                cf_attention.append([0] * n + [1] * len(ids))
                position = n + int(row_data["counterfactual_answer_position"])
                assert position == cf_maxlen - 1
                cf_positions.append(position)
                cf_source_indices.append(source_index)
                cf_pair_weights.append(float(row_data["counterfactual_pair_weight"]))
                cf_rule_ids.append(row_data["counterfactual_source_rule_id"])
            result.update(
                {
                    "counterfactual_input_ids": torch.tensor(cf_input_ids),
                    "counterfactual_attention_mask": torch.tensor(cf_attention),
                    "counterfactual_answer_positions": torch.tensor(cf_positions),
                    "counterfactual_source_indices": torch.tensor(cf_source_indices),
                    "counterfactual_pair_weights": torch.tensor(cf_pair_weights),
                    "counterfactual_source_rule_ids": cf_rule_ids,
                }
            )
        else:
            result.update(
                {
                    "counterfactual_input_ids": None,
                    "counterfactual_attention_mask": None,
                    "counterfactual_answer_positions": None,
                    "counterfactual_source_indices": None,
                    "counterfactual_pair_weights": None,
                    "counterfactual_source_rule_ids": [],
                }
            )
        return result

    def train_collate(batch):
        # FLA/Triton specialises kernels by tensor shape.  Rationale targets
        # range from ~50 to ~90 tokens, so dynamic batch maxima would trigger a
        # multi-minute compile for many distinct lengths.  One fixed padded
        # shape keeps the mathematical attention mask unchanged and compiles
        # exactly once per process/fold.
        return collate(
            batch,
            static_rationale_shape=args.rationale_weight > 0,
        )

    def model_output(batch):
        return model(input_ids=batch["input_ids"].to(device),
                     attention_mask=batch["attention_mask"].to(device))

    def verdict_logits(batch, out=None):
        if out is None:
            out = model_output(batch)
        row = torch.arange(out.logits.shape[0], device=device)
        pos = batch["answer_positions"].to(device)
        answer = out.logits[row, pos, :]
        return answer[:, yes_id], answer[:, no_id]

    def training_loss(batch):
        out = model_output(batch)
        ly, ln = verdict_logits(batch, out)
        # p(да) via softmax restricted to the two verdict tokens
        logp = F.log_softmax(torch.stack([ly, ln], dim=1), dim=1)
        p = batch["target_p"].to(device).float()
        loss_vec = _binary_verdict_ce(logp[:, 0], logp[:, 1], p)
        if batch.get("qpair_input_ids") is not None:
            pair_out = model(
                input_ids=batch["qpair_input_ids"].to(device),
                attention_mask=batch["qpair_attention_mask"].to(device),
            )
            pair_row = torch.arange(pair_out.logits.shape[0], device=device)
            pair_pos = batch["qpair_answer_positions"].to(device)
            pair_answer = pair_out.logits[pair_row, pair_pos, :]
            pair_logp = F.log_softmax(torch.stack([
                pair_answer[:, yes_id], pair_answer[:, no_id]
            ], dim=1), dim=1)
            source_indices = batch["qpair_source_indices"].to(device)
            pair_targets = p[source_indices]
            pair_ce = _binary_verdict_ce(
                pair_logp[:, 0], pair_logp[:, 1], pair_targets
            )
            loss_vec = loss_vec.clone()
            loss_vec[source_indices] = 0.5 * (
                loss_vec[source_indices] + pair_ce
            )
        w = batch["weight"].to(device).float() / weight_scale
        # Normalise by a FIXED dataset-level constant, not by this micro-batch's
        # own weight sum. Dividing by the local sum cancels the weights whenever
        # a micro-batch happens to be class-homogeneous, which at batch_size=2 is
        # almost always: measured on fold 3, the intended 17.7% of gradient mass
        # on rare positives collapsed to 3.1%, so the weights were nearly inert
        # and any experiment varying them compared 3.1% against 2.6%.
        verdict_loss = (loss_vec * w).mean()
        total = verdict_loss
        rationale_loss = None
        counterfactual_loss = None

        if args.rationale_weight > 0 and any(batch["aux_targets"]):
            # The verdict token is part of the input (label first).  For
            # rationale token k, the predicting logit is answer_position+k.
            aux_per_example = []
            aux_weights = []
            for i, targets in enumerate(batch["aux_targets"]):
                if not targets:
                    continue
                start = int(batch["answer_positions"][i]) + 1
                stop = start + len(targets)
                assert stop <= out.logits.shape[1] - 1
                assert batch["input_ids"][i, start + 1:stop + 1].tolist() == targets
                logits = out.logits[i, start:stop, :].float()
                target = torch.tensor(targets, device=device, dtype=torch.long)
                aux_per_example.append(
                    F.cross_entropy(logits, target, reduction="mean")
                )
                aux_weights.append(batch["aux_weight"][i].to(device).float())
            aux_losses = torch.stack(aux_per_example)
            rationale_loss = (
                (aux_losses * torch.stack(aux_weights)).sum()
                / (len(batch["aux_targets"]) * aux_mass_scale)
            )
            total = total + args.rationale_weight * rationale_loss

        if batch.get("rt_input_ids") is not None and batch["rt_targets"]:
            # Rationale as its own task: the sequence is prompt + slots, with no
            # verdict token, so nothing here is conditioned on the label.
            rt_out = model(input_ids=batch["rt_input_ids"].to(device),
                           attention_mask=batch["rt_attention_mask"].to(device))
            rt_losses, rt_w = [], []
            for i, targets in enumerate(batch["rt_targets"]):
                start = int(batch["rt_answer_positions"][i])
                stop = start + len(targets)
                logits = rt_out.logits[i, start:stop, :].float()
                target = torch.tensor(targets, device=device, dtype=torch.long)
                rt_losses.append(F.cross_entropy(logits, target, reduction="mean"))
                rt_w.append(batch["rt_weights"][i].to(device).float())
            rationale_loss = ((torch.stack(rt_losses) * torch.stack(rt_w)).sum()
                              / (len(batch["aux_targets"]) * aux_mass_scale))
            total = total + args.rationale_weight * rationale_loss

        if (
            args.counterfactual_weight > 0
            and batch["counterfactual_input_ids"] is not None
        ):
            if any(batch["aux_targets"]):
                raise AssertionError(
                    "counterfactual verdict mode must not contain rationale tokens"
                )
            counterfactual_out = model(
                input_ids=batch["counterfactual_input_ids"].to(device),
                attention_mask=batch["counterfactual_attention_mask"].to(device),
            )
            cf_row = torch.arange(
                counterfactual_out.logits.shape[0], device=device
            )
            cf_position = batch["counterfactual_answer_positions"].to(device)
            cf_answer = counterfactual_out.logits[cf_row, cf_position, :]
            cf_logp = F.log_softmax(
                torch.stack(
                    [cf_answer[:, yes_id], cf_answer[:, no_id]], dim=1
                ),
                dim=1,
            )
            # The synthetic target is directly the YES verdict token. There is
            # no generated explanation and no label token appended to the input.
            synthetic_yes_ce = _binary_verdict_ce(
                cf_logp[:, 0], cf_logp[:, 1], 1.0
            )
            source_indices = batch["counterfactual_source_indices"].to(device)
            if not torch.all(batch["labels_int"].to(device)[source_indices] == 0):
                raise AssertionError("paired real source must be a hard NO")
            pair_losses = _pair_average(
                loss_vec[source_indices], synthetic_yes_ce
            )
            pair_weights = batch["counterfactual_pair_weights"].to(device).float()
            if len(batch["counterfactual_source_rule_ids"]) != len(pair_losses):
                raise AssertionError("counterfactual rule ids lost during collation")
            counterfactual_loss = normalize_counterfactual_pair_losses(
                list(pair_losses.unbind()),
                list(pair_weights.unbind()),
                real_batch_size=len(batch["target_p"]),
                real_weight_scale=weight_scale,
            )
            total = total + args.counterfactual_weight * counterfactual_loss

        return (
            total,
            verdict_loss.detach(),
            None if rationale_loss is None else rationale_loss.detach(),
            None if counterfactual_loss is None else counterfactual_loss.detach(),
        )

    # Fixed normaliser so that E[w] = 1 over the training set: makes the
    # per-micro-batch mean an unbiased estimate of the weighted risk regardless
    # of how the batch happens to be composed.
    weight_scale = float(np.mean([r["weight"] * class_w[(r["category"], r["label"])]
                                  for r in train]))
    print(f"weight scale (mean example weight) = {weight_scale:.4f}")

    rationale_stratum_scale, rationale_stratum_stats = (
        compute_rationale_stratum_scales(train, rationale_targets, class_w)
        if args.rationale_weight > 0 else ({}, {})
    )
    aux_train_weights = [
        r["weight"] * class_w[(r["category"], r["label"])]
        * rationale_stratum_scale[(r["category"], r["label"])]
        for r in train if (args.rationale_weight > 0 and
                           int(r["id"]) in rationale_targets)
    ]
    aux_weight_scale = float(np.mean(aux_train_weights)) if aux_train_weights else 1.0
    aux_mass_scale = (float(sum(aux_train_weights) / len(train))
                      if aux_train_weights else 1.0)
    if args.rationale_weight > 0:
        printable_strata = {
            f"{category}|{label}": {
                key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in values.items()
            }
            for (category, label), values in rationale_stratum_stats.items()
        }
        print("rationale strata=" + json.dumps(
            printable_strata, ensure_ascii=False, sort_keys=True))
        print(f"rationale coverage={len(aux_train_weights)}/{len(train)} "
              f"aux weight scale={aux_weight_scale:.4f} "
              f"mass={aux_mass_scale:.4f}")

    train_ds = VerdictDataset(train, "train", with_aux=True)
    val_ds = (None if args.fixed_steps is not None else
              VerdictDataset(val, "val", with_aux=False))
    # Evaluation stores no activations, so it takes a much larger batch than
    # training — at the training batch size a single eval costs as much as
    # dozens of training steps.
    # Capped: no_grad removes stored activations, but this architecture's
    # fallback kernel still allocates large transients, so a huge eval batch
    # can OOM right after a training step that fit fine.
    eval_bs = args.eval_batch_size or min(16, max(4, args.batch_size * 4))

    @torch.no_grad()
    def evaluate():
        if val_ds is None:
            raise RuntimeError("fixed-step training must not evaluate the outer fold")
        model.eval()
        ps, ys, cats = [], [], []
        for batch in DataLoader(val_ds, batch_size=eval_bs, collate_fn=collate):
            ly, ln = verdict_logits(batch)
            p = torch.softmax(torch.stack([ly, ln], dim=1).float(), dim=1)[:, 0]
            ps += p.cpu().tolist()
            ys += batch["labels_int"].tolist()
            cats += batch["categories"]
        model.train()
        return tune_threshold_macro_f1(ys, ps, cats)

    if args.sampler_mode == LEGACY_SAMPLER_MODE:
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, collate_fn=train_collate, drop_last=False)
    else:
        from optimizer_window_sampler import FixedRunEpochSampler, sampler_stratum

        fixed_run_sampler = FixedRunEpochSampler(
            [sampler_stratum(record) for record in train],
            mode=args.sampler_mode,
            seed=args.seed,
            fixed_steps=args.fixed_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=fixed_run_sampler,
            shuffle=False, collate_fn=train_collate, drop_last=False,
        )
    if len(loader) == 0:
        raise RuntimeError("empty training loader after dataset construction")
    steps_total = (args.fixed_steps if args.fixed_steps is not None else
                   max(1, math.ceil(len(loader) * args.epochs / args.grad_accum)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(steps_total * 0.03), steps_total)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, step = -1.0, 0
    t0 = time.time()
    micro = 0
    print(f"starting: {steps_total} optimizer steps "
          f"({args.grad_accum} micro-batches of {args.batch_size} each)", flush=True)
    done = False
    window = []   # losses since the last log line — a single micro-batch at
    verdict_window, aux_window, counterfactual_window = [], [], []
    accum_in_step = 0
                  # batch_size 1 is one example, far too noisy to read a trend from
    while not done:
        for i, batch in enumerate(loader):
            (
                total_loss,
                verdict_loss,
                aux_loss,
                counterfactual_loss,
            ) = training_loss(batch)
            loss = total_loss / args.grad_accum
            loss.backward()
            window.append(float(total_loss.detach()))
            verdict_window.append(float(verdict_loss))
            if aux_loss is not None:
                aux_window.append(float(aux_loss))
            if counterfactual_loss is not None:
                counterfactual_window.append(float(counterfactual_loss))
            micro += 1
            accum_in_step += 1
            # Heartbeat during the first accumulation window, so a slow setup is
            # distinguishable from a hung one before the first optimizer step.
            if micro <= 3 or micro == args.grad_accum:
                print(f"  micro-batch {micro}/{args.grad_accum} "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)
            # Count accumulation globally, not with the per-epoch DataLoader
            # index. Otherwise an odd tail carries into the next epoch and the
            # first step there accidentally contains tail + grad_accum batches.
            if accum_in_step == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                accum_in_step = 0
                step += 1
                if step == 1:
                    # Triton kernels JIT-compile on the first pass (~70s). Timing
                    # the rest from here keeps that one-off out of the estimate,
                    # which would otherwise inflate the ETA by an order of magnitude.
                    t_steady = time.time()
                if step == 1 or step % args.log_every == 0:
                    el = time.time() - t0
                    eta = (0.0 if step <= 1 else
                           (time.time() - t_steady) / (step - 1) * (steps_total - step))
                    mean_loss = sum(window) / max(1, len(window))
                    mean_verdict = sum(verdict_window) / max(1, len(verdict_window))
                    aux_text = ("" if not aux_window else
                                f" | rationale {sum(aux_window) / len(aux_window):.4f}")
                    counterfactual_text = (
                        "" if not counterfactual_window else
                        " | counterfactual "
                        f"{sum(counterfactual_window) / len(counterfactual_window):.4f}"
                    )
                    print(f"step {step}/{steps_total} "
                          f"loss {mean_loss:.4f} (verdict {mean_verdict:.4f}"
                          f"{aux_text}{counterfactual_text}; avg of {len(window)}) | "
                          f"{el / 60:.1f}m elapsed" +
                          ("" if step <= 1 else f", ~{eta / 60:.0f}m left"), flush=True)
                    if mlflow:
                        mlflow.log_metric("loss", mean_loss, step=step)
                    window = []
                    verdict_window = []
                    aux_window = []
                    counterfactual_window = []
                if (args.fixed_steps is None and
                        (step % args.eval_every == 0 or step >= steps_total)):
                    honest_f1, optimistic_f1, thresholds, per_cat = evaluate()
                    print(f"step {step}: val macro F1 = {honest_f1:.4f} (honest) | "
                          f"{optimistic_f1:.4f} (tuned-on-self, optimistic)", flush=True)
                    for c, (h, o, n) in per_cat.items():
                        print(f"    {c:24s} honest {h:.4f} | optimistic {o:.4f} "
                              f"| threshold {thresholds[c]} | {n} positives", flush=True)
                    if mlflow:
                        mlflow.log_metric("val_macro_f1", honest_f1, step=step)
                        mlflow.log_metric("val_macro_f1_optimistic",
                                          optimistic_f1, step=step)
                    # Checkpoint selection follows the honest number — otherwise
                    # the most overfit step wins.
                    if honest_f1 > best_f1:
                        best_f1 = honest_f1
                        model.save_pretrained(out_dir / "best")
                        checkpoint_metadata = {"macro_f1": honest_f1,
                                        "macro_f1_optimistic": optimistic_f1,
                                        "thresholds": thresholds,
                                        "per_category": per_cat,
                                        "val_fold": args.val_fold,
                                        "outer_fold": args.val_fold,
                                        "outer_fold_untouched": False,
                                        "checkpoint_selection_fold": args.val_fold,
                                        "data_sha256": data_sha256,
                                        "training_config_sha256": training_config_sha256,
                                        "training_config": training_config,
                                        "counterfactual_provenance": (
                                            counterfactual_provenance
                                        ),
                                        "truncation": "description",
                                        "step": step, "args": vars(args)}
                        if backbone_runtime_provenance is not None:
                            checkpoint_metadata["backbone_runtime_provenance"] = (
                                backbone_runtime_provenance
                            )
                        (out_dir / "best" / "thresholds.json").write_text(
                            json.dumps(checkpoint_metadata, ensure_ascii=False))
                        print(f"  saved new best -> {out_dir / 'best'}")
                if step >= steps_total:
                    done = True
                    break
        else:
            continue

    if args.fixed_steps is not None:
        fixed_dir = out_dir / "fixed"
        model.save_pretrained(fixed_dir)
        checkpoint_metadata = {
            "thresholds": {},
            "val_fold": args.val_fold,
            "outer_fold": args.val_fold,
            "outer_fold_untouched": True,
            "checkpoint_selection_fold": None,
            "data_sha256": data_sha256,
            "training_config_sha256": training_config_sha256,
            "training_config": training_config,
            "counterfactual_provenance": counterfactual_provenance,
            "truncation": "description",
            "step": step,
            "args": vars(args),
        }
        if backbone_runtime_provenance is not None:
            checkpoint_metadata["backbone_runtime_provenance"] = (
                backbone_runtime_provenance
            )
        (fixed_dir / "thresholds.json").write_text(json.dumps(
            checkpoint_metadata, ensure_ascii=False))
        print(f"DONE. fixed-step adapter -> {fixed_dir}; outer fold was never evaluated")
    else:
        print(f"DONE. best honest val macro F1 = {best_f1:.4f} "
              f"(adapter in {out_dir / 'best'})")
    if mlflow:
        if args.fixed_steps is None:
            mlflow.log_metric("best_val_macro_f1", best_f1)
        mlflow.end_run()


if __name__ == "__main__":
    main()
