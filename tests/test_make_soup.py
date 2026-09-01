from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import make_soup  # noqa: E402


def _adapter(root: Path, name: str, a: torch.Tensor, b: torch.Tensor) -> Path:
    directory = root / name
    directory.mkdir()
    config = {
        "r": a.shape[0],
        "lora_alpha": a.shape[0] * 2,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
    }
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            "base_model.model.layer.q_proj.lora_A.weight": a,
            "base_model.model.layer.q_proj.lora_B.weight": b,
        },
        str(directory / "adapter_model.safetensors"),
    )
    return directory


def test_concatenation_equals_mean_delta(tmp_path):
    a1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b1 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    a2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    b2 = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    directories = [_adapter(tmp_path, "one", a1, b1), _adapter(tmp_path, "two", a2, b2)]
    output = tmp_path / "soup"
    report = make_soup.combine(directories, output)
    state = load_file(str(output / "adapter_model.safetensors"))
    a = state["base_model.model.layer.q_proj.lora_A.weight"].float()
    b = state["base_model.model.layer.q_proj.lora_B.weight"].float()
    assert torch.allclose(b @ a, (b1 @ a1 + b2 @ a2) / 2)
    assert report["rank"] == 4
    assert report["lora_alpha"] == 8


def test_incompatible_adapters_fail(tmp_path):
    one = _adapter(tmp_path, "one", torch.ones(2, 2), torch.ones(2, 2))
    two = _adapter(tmp_path, "two", torch.ones(1, 2), torch.ones(2, 1))
    try:
        make_soup.combine([one, two], tmp_path / "out")
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("incompatible adapters were accepted")

