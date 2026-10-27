import glob
import json
import os
from typing import Optional
from dataclasses import dataclass


def resolve_model_path(path: str) -> str:
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snaps = sorted(glob.glob(os.path.join(path, "snapshots", "*")), key=os.path.getmtime)
    for snap in reversed(snaps):  # 有多个版本时取最新的
        if os.path.isfile(os.path.join(snap, "config.json")):
            return snap
    raise FileNotFoundError(f"{path} 下找不到 config.json（也没有 snapshots/*/config.json）")

@dataclass
class EngineConfig:
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    max_model_len: int = 4096
    num_gpu_blocks: Optional[int] = None
    attn_backend: str = "flashinfer"
    enforce_eager: bool = True
    max_graph_bs: int = 256


    def __post_init__(self):
        assert self.max_model_len <= self.max_num_batched_tokens, (f"max_model_len： {self.max_model_len} "
                                                         f"必须小于等于 max_num_batched_tokens: {self.max_num_batched_tokens}")
        assert self.max_num_seqs <= self.max_num_batched_tokens, (f"max_num_seqs: {self.max_num_seqs} 必须小于等于 max_num_batched_tokens"
                                                        f": {self.max_num_batched_tokens}")
        assert self.max_graph_bs <= self.max_num_seqs

@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool
    head_dim: int
    # 引擎字段：RoPE cos/sin 表预计算的长度，也是单条序列（prompt + 生成）的上限
    max_model_len: int = 4096

    def __post_init__(self):
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "GQA 要求 q 头数是 kv 头数的整数倍"
        assert self.head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"


def load_model_config(model_path: str, max_model_len: int = 4096) -> ModelConfig:
    model_path = resolve_model_path(model_path)
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)

    # Qwen2.5 的 config.json 没有 head_dim，按 hidden_size / num_heads 算
    # （1.5B：1536 / 12 = 128）。若以后换 Qwen3，它显式给了 head_dim 且不满足这个公式，
    # 所以优先读字段。
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]

    # 不能超过模型训练时支持的位置数
    max_pos = cfg.get("max_position_embeddings")
    if max_pos is not None:
        max_model_len = min(max_model_len, max_pos)

    return ModelConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        vocab_size=cfg["vocab_size"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        head_dim=head_dim,
        max_model_len=max_model_len,
    )

def load_generation_config(model_path: str) -> list[int]:
    model_path = os.path.join(resolve_model_path(model_path), "generation_config.json")
    with open(model_path) as f:
        generation_cfg = json.load(f)

    if isinstance(generation_cfg["eos_token_id"], int):
        return [generation_cfg["eos_token_id"]]
    else:
        return generation_cfg["eos_token_id"]
