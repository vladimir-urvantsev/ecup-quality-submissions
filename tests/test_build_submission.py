from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_submission  # noqa: E402


def test_public_build_sanitizes_model_path_and_pins_thresholds(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 96,
                "lora_alpha": 192,
                "base_model_name_or_path": "/private/machine/path",
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = build_submission.build(ROOT, adapter, ROOT / "configs/recipe.json", output)
    public_config = json.loads((output / "adapter/adapter_config.json").read_text())
    thresholds = json.loads((output / "adapter/thresholds.json").read_text())
    assert public_config["base_model_name_or_path"] == "Qwen/Qwen3.5-4B"
    assert thresholds["thresholds"] == {"БАД": 0.2123, "Легковоспламеняющиеся": 0.717}
    assert report["text_only"] is True
    assert report["rationale_is_post_verdict"] is True

