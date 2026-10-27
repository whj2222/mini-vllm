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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")

@pytest.fixture(scope="module")
def model() -> Qwen2ForCausalLM:
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        config = load_model_config(MODEL_PATH)
        with torch.device("cuda"):
            model = Qwen2ForCausalLM(config)
        load_weights(model, MODEL_PATH)
    finally:
        torch.set_default_dtype(old)
    return model

@pytest.fixture(scope="module")
def weight_file() -> list[str]:
    model_dir = resolve_model_path(MODEL_PATH)
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    assert files, f"{model_dir} 下没有 *.safetensors"
    return files