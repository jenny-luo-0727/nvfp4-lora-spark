"""load_non_nvfp4_weights keeps a buffer in FP32 when the model wants it in FP32.

A router's `e_score_correction_bias` is the case that matters: its per-expert differences are
finer than BF16's spacing at its magnitude, so the blanket BF16 cast used to collapse the values and
change the top-k expert selection. A model asks for FP32 in one of two ways:
  - remote-code Nemotron-H (e.g. Nemotron 3 Super) registers the buffer in FP32 and the checkpoint
    stores it in FP32;
  - transformers' native Nemotron-H and GLM-4-MoE list it in `_keep_in_fp32_modules_strict`.
Parameters, and buffers the model asks for in neither way, still follow the requested dtype.

Everything here runs on CPU with tiny synthetic safetensors shards.
"""
from __future__ import annotations

import json
import types

import torch
import torch.nn as nn
from safetensors.torch import save_file

from nvfp4_lora.loader import load_non_nvfp4_weights

CPU = torch.device("cpu")
BIAS_KEY = "model.language_model.gate.e_score_correction_bias"


class _TinyRouterShaped(nn.Module):
    """qwen3_5-translated paths (`model.language_model.X` -> `model.X`) with a router: a BF16 gate
    weight, a correction-bias buffer registered in `bias_dtype`, and a BF16 buffer."""

    def __init__(self, bias_dtype=torch.float32, keep_strict=None):
        super().__init__()
        self.model = nn.Module()
        self.model.gate = nn.Module()
        self.model.gate.weight = nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16))
        self.model.gate.register_buffer("e_score_correction_bias", torch.zeros(8, dtype=bias_dtype))
        self.model.gate.register_buffer("lowp_buffer", torch.zeros(8, dtype=torch.bfloat16))
        self.lm_head = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.config = types.SimpleNamespace(model_type="qwen3_5_moe_text")
        if keep_strict is not None:
            self._keep_in_fp32_modules_strict = set(keep_strict)


def _write_checkpoint(tmp_path, bias: torch.Tensor):
    tensors = {
        "model.language_model.gate.weight": torch.randn(8, 4),
        BIAS_KEY: bias,
        "model.language_model.gate.lowp_buffer": torch.randn(8),
        "lm_head.weight": torch.randn(4, 4),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    index = {"weight_map": {k: shard for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def _router_bias():
    # Large mean, small spread. BF16 has 8 bits of precision (7 stored), so at ~12 its spacing is
    # 0.0625 and every value below rounds to 12.0.
    torch.manual_seed(0)
    return 12.0 + 0.005 * torch.randn(8)


def _load(tmp_path, model):
    load_non_nvfp4_weights(model, tmp_path, device=CPU, dtype=torch.bfloat16, strict=True)
    return model.model.gate.e_score_correction_bias


def test_fp32_registered_buffer_stays_fp32_bit_exact(tmp_path):
    bias = _router_bias()
    assert len(bias.to(torch.bfloat16).unique()) == 1  # the collapse this guards against
    _write_checkpoint(tmp_path, bias)
    model = _TinyRouterShaped(bias_dtype=torch.float32)
    loaded = _load(tmp_path, model)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, bias)
    assert "e_score_correction_bias" in dict(model.model.gate.named_buffers(recurse=False))


def test_keep_in_fp32_strict_buffer_loads_fp32_bit_exact(tmp_path):
    # transformers' native Nemotron-H registers the buffer without a dtype (BF16 under a BF16 build)
    # and asks for FP32 through `_keep_in_fp32_modules_strict` instead.
    bias = _router_bias()
    _write_checkpoint(tmp_path, bias)
    model = _TinyRouterShaped(bias_dtype=torch.bfloat16, keep_strict={"e_score_correction_bias"})
    loaded = _load(tmp_path, model)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, bias)


def test_keep_in_fp32_strict_upcasts_bf16_on_disk(tmp_path):
    # Same as from_pretrained: a listed module is upcast to FP32 whatever the checkpoint stores.
    bias = _router_bias().to(torch.bfloat16)
    _write_checkpoint(tmp_path, bias)
    model = _TinyRouterShaped(bias_dtype=torch.bfloat16, keep_strict={"e_score_correction_bias"})
    loaded = _load(tmp_path, model)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, bias.float())


def test_fp32_slot_alone_does_not_upcast_bf16_on_disk(tmp_path):
    # A model built at the default FP32 dtype has every slot in FP32, so an unlisted FP32 slot is
    # not enough on its own: the checkpoint must also store the tensor in FP32.
    _write_checkpoint(tmp_path, _router_bias().to(torch.bfloat16))
    model = _TinyRouterShaped(bias_dtype=torch.float32)
    assert _load(tmp_path, model).dtype == torch.bfloat16


def test_parameters_and_unlisted_lowp_buffers_still_follow_requested_dtype(tmp_path):
    _write_checkpoint(tmp_path, _router_bias())
    model = _TinyRouterShaped(bias_dtype=torch.bfloat16, keep_strict={"e_score_correction_bias"})
    _load(tmp_path, model)
    assert model.model.gate.weight.dtype == torch.bfloat16
    assert model.model.gate.lowp_buffer.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
