import glob
import os

import torch
import torch.nn as nn
from safetensors import safe_open

from mini_vllm.config import resolve_model_path

def load_weights(model: nn.Module, model_path: str):
    model_path = resolve_model_path(model_path)
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{model_path} 下没有 *.safetensors 文件")
    params = dict(model.named_parameters())
    loaded = set()
    unexpected = set()
    with torch.no_grad():
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name not in params:
                        unexpected.add(name)
                        continue
                    tensor = f.get_tensor(name)
                    param = params[name]

                    if param.shape != tensor.shape:
                        raise ValueError(f"{name} 形状不匹配：模型 {tuple(param.shape)} 文件{tuple(tensor.shape)}")
                    param.copy_(tensor)
                    loaded.add(name)
    missing = sorted(set(params) - loaded)
    unexpected = sorted(unexpected)
    errors = []
    if missing:
        errors.append(f"模型里有、文件里没有（共 {len(missing)} 个）：{missing[:10]}")
    if unexpected:
        errors.append(f"文件里有、模型里没有（共 {len(unexpected)} 个）：{unexpected[:10]}")
    if errors:
        raise RuntimeError("权重加载失败\n" + "\n".join(errors))