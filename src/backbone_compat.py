"""Fail-closed compatibility contract for opt-in classifier backbones.

The production recipe is still Qwen3.5-4B.  This module exists only for a
matched, text-only ``google/gemma-4-E4B-it`` diversity experiment.  Gemma 4 is
a multimodal checkpoint in Transformers, so a suffix-only PEFT target such as
``q_proj`` would also match modules outside the language model.  The contract
below instead resolves every permitted full module path, audits the resulting
PEFT wrapper, and records model/tokenizer/package provenance.

The module imports neither torch nor transformers at import time.  This keeps
ordinary Qwen dry-runs and the existing test suite byte/side-effect compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


LEGACY_QWEN_PROFILE = "qwen35_4b_legacy"
GEMMA4_E4B_PROFILE = "gemma4_e4b_text_lora_v1"
BACKBONE_PROFILES = (LEGACY_QWEN_PROFILE, GEMMA4_E4B_PROFILE)
BACKBONE_CONTRACT_VERSION = "classifier-backbone-compat-v1"
PREFLIGHT_SCHEMA_VERSION = "gemma4-e4b-local-preflight-v1"
RUNTIME_SCHEMA_VERSION = "gemma4-e4b-runtime-audit-v1"

GEMMA4_EXPECTED_MODEL_TYPE = "gemma4"
GEMMA4_EXPECTED_ARCHITECTURE = "Gemma4ForConditionalGeneration"
GEMMA4_MIN_TRANSFORMERS = "5.5.0"
GEMMA4_E4B_TEXT_SHAPE = {
    "num_hidden_layers": 42,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "vocab_size": 262144,
    "num_attention_heads": 8,
    "num_kv_shared_layers": 18,
}
VERDICT_YES = "да"
VERDICT_NO = "нет"

# Full-match only.  In particular, this cannot select vision_tower,
# audio_tower, embed_vision, embed_audio, router, experts, lm_head, embeddings,
# or the per-layer input projections.
GEMMA4_LANGUAGE_LORA_PATTERN = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
GEMMA4_MANDATORY_PER_LAYER = (
    "self_attn.q_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class BackboneContractError(ValueError):
    """Raised before training when a backbone contract is not provable."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_versions() -> dict[str, str | None]:
    return {
        "python": ".".join(map(str, os.sys.version_info[:3])),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "accelerate": _version("accelerate"),
        "safetensors": _version("safetensors"),
    }


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", str(value))
    if not match:
        raise BackboneContractError(f"cannot parse package version {value!r}")
    return tuple(int(piece) for piece in match.group(1).split("."))


def require_gemma4_transformers_version(version: str | None = None) -> str:
    version = version or _version("transformers")
    if version is None:
        raise BackboneContractError("transformers is not installed")
    actual = _version_tuple(version)
    required = _version_tuple(GEMMA4_MIN_TRANSFORMERS)
    width = max(len(actual), len(required))
    actual += (0,) * (width - len(actual))
    required += (0,) * (width - len(required))
    if actual < required:
        raise BackboneContractError(
            f"Gemma 4 requires transformers>={GEMMA4_MIN_TRANSFORMERS}; got {version}"
        )
    return version


def validate_profile(profile: str) -> str:
    profile = str(profile)
    if profile not in BACKBONE_PROFILES:
        raise BackboneContractError(
            f"unknown backbone profile {profile!r}; expected one of {BACKBONE_PROFILES}"
        )
    return profile


def resolve_backbone_profile(
    explicit: str | None,
    *,
    checkpoint_args: Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
) -> str:
    stored = []
    for source in (checkpoint_args or {}, training_config or {}):
        value = source.get("backbone_profile")
        if value not in (None, ""):
            stored.append(validate_profile(str(value)))
    if len(set(stored)) > 1:
        raise BackboneContractError(
            f"checkpoint contains conflicting backbone profiles: {stored}"
        )
    explicit_value = None if explicit in (None, "") else validate_profile(explicit)
    checkpoint_value = stored[0] if stored else LEGACY_QWEN_PROFILE
    if explicit_value is not None and explicit_value != checkpoint_value:
        raise BackboneContractError(
            f"explicit backbone profile {explicit_value!r} conflicts with "
            f"checkpoint profile {checkpoint_value!r}"
        )
    return explicit_value or checkpoint_value


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackboneContractError(f"cannot read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise BackboneContractError(f"{context} must contain a JSON object: {path}")
    return value


def _gemma4_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("model_type") != GEMMA4_EXPECTED_MODEL_TYPE:
        raise BackboneContractError(
            f"expected model_type={GEMMA4_EXPECTED_MODEL_TYPE!r}, got "
            f"{config.get('model_type')!r}"
        )
    architectures = config.get("architectures")
    if architectures != [GEMMA4_EXPECTED_ARCHITECTURE]:
        raise BackboneContractError(
            f"expected architectures=[{GEMMA4_EXPECTED_ARCHITECTURE!r}], got "
            f"{architectures!r}"
        )
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise BackboneContractError("Gemma 4 checkpoint has no text_config object")
    try:
        layers = int(text["num_hidden_layers"])
        hidden_size = int(text["hidden_size"])
        vocab_size = int(text["vocab_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackboneContractError(
            "Gemma 4 text_config lacks integer layer/hidden/vocab dimensions"
        ) from exc
    if min(layers, hidden_size, vocab_size) <= 0:
        raise BackboneContractError("Gemma 4 text dimensions must be positive")
    shape_drift = {
        key: (text.get(key), expected)
        for key, expected in GEMMA4_E4B_TEXT_SHAPE.items()
        if text.get(key) != expected
    }
    if shape_drift:
        raise BackboneContractError(
            f"checkpoint is not the expected Gemma-4-E4B text shape: {shape_drift}"
        )
    if text.get("enable_moe_block") is not False or text.get("num_experts") is not None:
        raise BackboneContractError("Gemma-4-E4B profile expects the dense text backbone")
    return {
        "model_type": config["model_type"],
        "architecture": architectures[0],
        "transformers_version_declared": config.get("transformers_version"),
        "text_num_hidden_layers": layers,
        "text_hidden_size": hidden_size,
        "text_vocab_size": vocab_size,
        "text_intermediate_size": int(text["intermediate_size"]),
        "text_num_attention_heads": int(text["num_attention_heads"]),
        "text_num_kv_shared_layers": int(text["num_kv_shared_layers"]),
        "text_enable_moe_block": bool(text["enable_moe_block"]),
        "vision_tower_declared": isinstance(config.get("vision_config"), dict),
        "audio_tower_declared": isinstance(config.get("audio_config"), dict),
    }


def _weight_inventory(root: Path, index: Mapping[str, Any] | None) -> list[Path]:
    if index is not None:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise BackboneContractError("model.safetensors.index.json has no weight_map")
        names = sorted({str(name) for name in weight_map.values()})
    else:
        names = ["model.safetensors"]
    paths = []
    for name in names:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise BackboneContractError(f"missing/empty weight shard: {path}")
        paths.append(path)
    return paths


def static_model_provenance(
    model_path: str | Path,
    profile: str,
    *,
    hash_weights: bool,
) -> dict[str, Any]:
    """Hash the local model contract without loading any model weights."""

    profile = validate_profile(profile)
    if profile != GEMMA4_E4B_PROFILE:
        raise BackboneContractError("static provenance is opt-in Gemma-only")
    require_gemma4_transformers_version()
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise BackboneContractError(f"model directory does not exist: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise BackboneContractError(f"missing model config: {config_path}")
    config = _read_json(config_path, context="model config")
    config_contract = _gemma4_config_contract(config)
    index_path = root / "model.safetensors.index.json"
    index = _read_json(index_path, context="model index") if index_path.is_file() else None
    weights = _weight_inventory(root, index)
    weight_rows = []
    for path in weights:
        row: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
        if hash_weights:
            row["sha256"] = sha256_file(path)
        weight_rows.append(row)

    small_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    )
    small_hashes = {
        name: sha256_file(root / name)
        for name in small_names
        if (root / name).is_file()
    }
    snapshot_revision = None
    if root.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{7,64}", root.name):
        snapshot_revision = root.name
    result = {
        "profile": profile,
        "resolved_path": str(root),
        **config_contract,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
        "small_artifact_sha256": small_hashes,
        "snapshot_revision": snapshot_revision,
        "weights_content_hashed": bool(hash_weights),
        "weight_shards": weight_rows,
        "weight_inventory_sha256": stable_json_sha256(weight_rows),
    }
    return result


def single_token_verdict_contract(tokenizer) -> dict[str, Any]:
    tokens = {}
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    for role, text in (("yes", VERDICT_YES), ("no", VERDICT_NO)):
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) != 1:
            raise BackboneContractError(
                f"Gemma verdict {text!r} must be exactly one token, got {ids}"
            )
        token_id = int(ids[0])
        if token_id < 0 or token_id in special_ids:
            raise BackboneContractError(
                f"Gemma verdict {text!r} resolved to invalid/special token {token_id}"
            )
        tokens[role] = {"text": text, "token_id": token_id}
    if tokens["yes"]["token_id"] == tokens["no"]["token_id"]:
        raise BackboneContractError("yes/no verdicts resolved to the same token")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise BackboneContractError("Gemma tokenizer has no pad_token_id")
    if int(pad_id) in {tokens["yes"]["token_id"], tokens["no"]["token_id"]}:
        raise BackboneContractError("pad token collides with a verdict token")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise BackboneContractError("Gemma tokenizer has no non-empty chat_template")
    messages = [
        {"role": "system", "content": "Ты — модератор маркетплейса."},
        {"role": "user", "content": "Ответь одним словом: да или нет."},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if not isinstance(rendered, str) or not rendered:
        raise BackboneContractError("Gemma chat template rendered an empty prompt")
    rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not rendered_ids:
        raise BackboneContractError("Gemma chat template tokenized to an empty prompt")
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(len(tokenizer)),
        "pad_token_id": int(pad_id),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "rendered_probe_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "rendered_probe_tokens": len(rendered_ids),
        "verdicts": tokens,
    }


def load_tokenizer_for_profile(model_path: str | Path, profile: str):
    profile = validate_profile(profile)
    if profile != GEMMA4_E4B_PROFILE:
        raise BackboneContractError("opt-in tokenizer loader is Gemma-only")
    require_gemma4_transformers_version()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    contract = single_token_verdict_contract(tokenizer)
    return tokenizer, contract


def write_preflight(
    model_path: str | Path,
    profile: str,
    out: str | Path,
    *,
    hash_weights: bool,
) -> dict[str, Any]:
    model = static_model_provenance(
        model_path, profile, hash_weights=hash_weights
    )
    _tokenizer, tokenizer_contract = load_tokenizer_for_profile(model_path, profile)
    result = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PASS",
        "created_unix": time.time(),
        "contract_version": BACKBONE_CONTRACT_VERSION,
        "profile": profile,
        "model": model,
        "tokenizer": tokenizer_contract,
        "environment": environment_versions(),
        "code_sha256": sha256_file(Path(__file__)),
        "language_lora_fullmatch_regex": GEMMA4_LANGUAGE_LORA_PATTERN.pattern,
        "weights_content_hashed": bool(hash_weights),
    }
    result["contract_sha256"] = stable_json_sha256(
        {key: value for key, value in result.items() if key != "created_unix"}
    )
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise BackboneContractError(f"refusing to overwrite preflight: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return result


def validate_preflight_file(
    path: str | Path,
    model_path: str | Path,
    profile: str,
    *,
    require_weight_hashes: bool = True,
) -> dict[str, Any]:
    value = _read_json(Path(path), context="backbone preflight")
    if value.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise BackboneContractError("unexpected Gemma preflight schema")
    if value.get("status") != "PASS" or value.get("profile") != profile:
        raise BackboneContractError("Gemma preflight did not PASS this profile")
    if value.get("contract_version") != BACKBONE_CONTRACT_VERSION:
        raise BackboneContractError("Gemma preflight contract version drift")
    if value.get("code_sha256") != sha256_file(Path(__file__)):
        raise BackboneContractError("Gemma backbone compatibility code drift")
    expected_contract_sha = stable_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_unix", "contract_sha256"}
    })
    if value.get("contract_sha256") != expected_contract_sha:
        raise BackboneContractError("Gemma preflight contract SHA is invalid")
    model = value.get("model")
    if not isinstance(model, dict):
        raise BackboneContractError("Gemma preflight has no model provenance")
    if model.get("resolved_path") != str(Path(model_path).resolve()):
        raise BackboneContractError("Gemma preflight model path mismatch")
    if require_weight_hashes and not model.get("weights_content_hashed"):
        raise BackboneContractError("Gemma preflight did not hash model weights")
    for row in model.get("weight_shards", []):
        if require_weight_hashes and not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            raise BackboneContractError("Gemma preflight has an unhashed weight shard")
    current = static_model_provenance(model_path, profile, hash_weights=False)
    stable_keys = (
        "resolved_path",
        "model_type",
        "architecture",
        "config_sha256",
        "index_sha256",
        "small_artifact_sha256",
        "text_num_hidden_layers",
        "text_hidden_size",
        "text_vocab_size",
    )
    drift = {
        key: (model.get(key), current.get(key))
        for key in stable_keys
        if model.get(key) != current.get(key)
    }
    recorded_sizes = [
        (row.get("name"), int(row.get("bytes", -1)))
        for row in model.get("weight_shards", [])
    ]
    current_sizes = [
        (row.get("name"), int(row.get("bytes", -1)))
        for row in current.get("weight_shards", [])
    ]
    if recorded_sizes != current_sizes:
        drift["weight_shard_sizes"] = (recorded_sizes, current_sizes)
    if drift:
        raise BackboneContractError(f"Gemma model drift since preflight: {drift}")
    return value


def validate_tokenizer_against_preflight(
    tokenizer_contract: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    recorded = preflight.get("tokenizer")
    if not isinstance(recorded, dict):
        raise BackboneContractError("Gemma preflight has no tokenizer contract")
    if dict(tokenizer_contract) != recorded:
        raise BackboneContractError("Gemma tokenizer drifted since preflight")


def load_model_for_profile(model_path: str | Path, profile: str, *, dtype):
    profile = validate_profile(profile)
    if profile != GEMMA4_E4B_PROFILE:
        raise BackboneContractError("opt-in model loader is Gemma-only")
    require_gemma4_transformers_version()
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if getattr(config, "model_type", None) != GEMMA4_EXPECTED_MODEL_TYPE:
        raise BackboneContractError(
            f"AutoConfig resolved unexpected model_type={getattr(config, 'model_type', None)!r}"
        )
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=dtype, **kwargs
        )
    if model.__class__.__name__ != GEMMA4_EXPECTED_ARCHITECTURE:
        raise BackboneContractError(
            f"AutoModelForCausalLM resolved {model.__class__.__name__!r}, expected "
            f"{GEMMA4_EXPECTED_ARCHITECTURE!r}"
        )
    try:
        layers = model.model.language_model.layers
    except AttributeError as exc:
        raise BackboneContractError(
            "Gemma model lacks model.language_model.layers"
        ) from exc
    if len(layers) != int(config.text_config.num_hidden_layers):
        raise BackboneContractError("Gemma runtime layer count differs from config")
    return model


def disable_model_cache(model, profile: str) -> None:
    profile = validate_profile(profile)
    if hasattr(model, "config"):
        model.config.use_cache = False
        if profile == GEMMA4_E4B_PROFILE and hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False


def resolve_gemma4_lora_targets(model) -> tuple[list[str], dict[str, Any]]:
    """Return full paths for text attention/MLP Linear modules only."""

    import torch

    modules = dict(model.named_modules())
    targets = sorted(
        name
        for name, module in modules.items()
        if GEMMA4_LANGUAGE_LORA_PATTERN.fullmatch(name)
        and isinstance(module, torch.nn.Linear)
    )
    if not targets:
        raise BackboneContractError("Gemma language-model LoRA regex matched no Linear modules")
    try:
        n_layers = int(model.config.text_config.num_hidden_layers)
    except (AttributeError, TypeError, ValueError) as exc:
        raise BackboneContractError("cannot resolve Gemma text layer count") from exc
    target_set = set(targets)
    missing = []
    expected = set()
    text_config = model.config.text_config
    shared_start = n_layers - int(text_config.num_kv_shared_layers)
    layer_types = list(text_config.layer_types)
    for layer in range(n_layers):
        prefix = f"model.language_model.layers.{layer}."
        for suffix in GEMMA4_MANDATORY_PER_LAYER:
            path = prefix + suffix
            expected.add(path)
            if path not in target_set:
                missing.append(path)
        if layer < shared_start:
            expected.add(prefix + "self_attn.k_proj")
            use_alternative_attention = bool(text_config.attention_k_eq_v) and (
                layer_types[layer] != "sliding_attention"
            )
            if not use_alternative_attention:
                expected.add(prefix + "self_attn.v_proj")
    if missing:
        raise BackboneContractError(
            f"Gemma language LoRA inventory misses {len(missing)} mandatory modules; "
            f"first={missing[:5]}"
        )
    if target_set != expected:
        raise BackboneContractError(
            "Gemma attention/MLP inventory differs from config: "
            f"missing={sorted(expected - target_set)[:10]} "
            f"extra={sorted(target_set - expected)[:10]}"
        )
    forbidden = [
        path
        for path in targets
        if any(piece in path for piece in (
            "vision_tower", "audio_tower", ".experts.", ".router.",
            "embed_vision", "embed_audio", "lm_head", "embed_tokens",
        ))
    ]
    if forbidden:
        raise BackboneContractError(f"forbidden Gemma LoRA targets: {forbidden[:5]}")
    per_suffix: dict[str, int] = {}
    for path in targets:
        suffix = re.sub(
            r"^model\.language_model\.layers\.\d+\.", "", path
        )
        per_suffix[suffix] = per_suffix.get(suffix, 0) + 1
    audit = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "profile": GEMMA4_E4B_PROFILE,
        "model_class": model.__class__.__name__,
        "config_class": model.config.__class__.__name__,
        "model_type": getattr(model.config, "model_type", None),
        "text_num_hidden_layers": n_layers,
        "target_regex": GEMMA4_LANGUAGE_LORA_PATTERN.pattern,
        "target_count": len(targets),
        "target_paths_sha256": stable_json_sha256(targets),
        "target_counts_by_suffix": dict(sorted(per_suffix.items())),
        "target_paths": targets,
    }
    return targets, audit


def _strip_peft_prefix(path: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def audit_peft_lora_scope(peft_model, expected_targets: Iterable[str]) -> dict[str, Any]:
    expected = set(map(str, expected_targets))
    actual = set()
    for name, module in peft_model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and len(lora_a):
            actual.add(_strip_peft_prefix(name))
    if actual != expected:
        raise BackboneContractError(
            "PEFT Gemma LoRA scope mismatch: "
            f"missing={sorted(expected - actual)[:10]} extra={sorted(actual - expected)[:10]}"
        )
    trainable = [name for name, parameter in peft_model.named_parameters() if parameter.requires_grad]
    forbidden = [name for name in trainable if ".lora_" not in name]
    if forbidden:
        raise BackboneContractError(
            f"non-LoRA Gemma parameters are trainable; first={forbidden[:10]}"
        )
    return {
        "actual_target_count": len(actual),
        "actual_target_paths_sha256": stable_json_sha256(sorted(actual)),
        "trainable_parameter_tensors": len(trainable),
        "all_trainable_parameters_are_lora": True,
    }


def adapter_lora_paths(adapter_dir: str | Path) -> tuple[set[str], dict[str, set[str]]]:
    from safetensors import safe_open

    adapter_dir = Path(adapter_dir)
    state_path = adapter_dir / "adapter_model.safetensors"
    pairs: dict[str, set[str]] = {}
    with safe_open(str(state_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    for key in keys:
        if ".lora_A" not in key and ".lora_B" not in key:
            continue
        which = "A" if ".lora_A" in key else "B"
        path = _strip_peft_prefix(key.split(".lora_")[0])
        pairs.setdefault(path, set()).add(which)
    incomplete = {path: value for path, value in pairs.items() if value != {"A", "B"}}
    if incomplete:
        raise BackboneContractError(f"incomplete LoRA A/B pairs: {incomplete}")
    return set(pairs), pairs


def validate_adapter_scope(adapter_dir: str | Path, expected_targets: Iterable[str]) -> dict[str, Any]:
    expected = set(map(str, expected_targets))
    actual, _pairs = adapter_lora_paths(adapter_dir)
    if actual != expected:
        raise BackboneContractError(
            "saved Gemma adapter scope mismatch: "
            f"missing={sorted(expected - actual)[:10]} extra={sorted(actual - expected)[:10]}"
        )
    config = _read_json(Path(adapter_dir) / "adapter_config.json", context="adapter config")
    configured = set(map(str, config.get("target_modules") or []))
    if configured != expected:
        raise BackboneContractError(
            "Gemma adapter_config target_modules differs from runtime inventory"
        )
    return {
        "adapter_target_count": len(actual),
        "adapter_target_paths_sha256": stable_json_sha256(sorted(actual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--model_path", required=True)
    preflight.add_argument(
        "--profile", choices=(GEMMA4_E4B_PROFILE,), required=True
    )
    preflight.add_argument("--out", required=True)
    preflight.add_argument("--hash_weights", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        report = write_preflight(
            args.model_path,
            args.profile,
            args.out,
            hash_weights=args.hash_weights,
        )
        print(json.dumps({
            "status": report["status"],
            "profile": report["profile"],
            "model": report["model"]["resolved_path"],
            "weight_shards": len(report["model"]["weight_shards"]),
            "weights_content_hashed": report["model"]["weights_content_hashed"],
            "contract_sha256": report["contract_sha256"],
            "out": str(Path(args.out).resolve()),
        }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
