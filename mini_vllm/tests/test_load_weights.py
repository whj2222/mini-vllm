import pytest
import torch
import glob
import os
import torch.nn as nn
from safetensors import safe_open

from mini_vllm.config import load_model_config, resolve_model_path
from mini_vllm.models.qwen2 import Qwen2ForCausalLM
from mini_vllm.layers.layernorm import RMSNorm
from mini_vllm.utils.loader import load_weights

MODEL_PATH = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct"

# ---------- 正常加载 ----------
def test_param_name_match_file(model, weight_file):
    model_keys = set(dict(model.named_parameters()))
    file_names = set()
    for file in weight_file:
        with safe_open(file, "pt") as f:
            file_names.update(f.keys())
    assert file_names == model_keys

def test_tied_embeddings(model):
    assert model.lm_head.weight is model.model.embed_tokens.weight

def test_values_match_file(model, weight_file):
    CHECK_NAMES = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
        "model.norm.weight",
    ]
    params = dict(model.named_parameters())
    checks = set()
    for file in weight_file:
        with safe_open(file, "pt") as f:
            keys = set(f.keys())
            for name in CHECK_NAMES:
                if name not in keys:
                    continue
                tensor = f.get_tensor(name).to("cuda")
                assert torch.equal(params[name], tensor)
                checks.add(name)
    assert checks == set(CHECK_NAMES)

def test_dtype_and_device(model):
    params = model.named_parameters()
    for name, param in params:
        assert param.dtype == torch.bfloat16
        assert param.device.type == "cuda"

class Wrong(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

def test_unexpected_and_missing_detected():
    with pytest.raises(RuntimeError) as exc_info:
        load_weights(Wrong(), MODEL_PATH)
    msg = str(exc_info.value)
    assert "模型里有、文件里没有" in msg
    assert "文件里有、模型里没有" in msg
    assert "linear.weight" in msg

class Wrong_shape(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.norm = RMSNorm(10)

def test_shape_mismatch_detected():
    with pytest.raises(ValueError) as exc_info:
        load_weights(Wrong_shape(), MODEL_PATH)
    assert "model.norm.weight" in str(exc_info.value)