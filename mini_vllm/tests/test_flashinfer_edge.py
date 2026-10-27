"""flashinfer 边界回归：一行的合法分数全部极负时不能输出 0。

背景（2026-10-16 定位）：flashinfer 0.6.18 的 "-inf" 是 -5e4（include/flashinfer/math.cuh:33），
prefill kernel 用它初始化 running max，并且按**未乘 sm_scale 的 q·k** 比较；一行原始分数全低于 -5e4
就被当成整行 mask，输出 0。Qwen2.5 第 0 层 |k|≈320，第 3 个头的少数 token 原始分数到 -6 万，
导致 flashinfer 后端比 torch 后端和 fp32 真值少对 8 个 token。
修法：Attention 里 q 先乘 1/sqrt(D)（fi_q_scale），plan 传 sm_scale=1.0，阈值移到缩放后分数 -5e4。

这里直接调 Attention._prefill_attn / _decode_attn（真实代码路径），用合成数据让每行分数都等于 s。
运行（项目根目录）：python -m pytest mini_vllm/tests/test_flashinfer_edge.py -v
"""
import types

import pytest
import torch
import torch.nn.functional as F

from mini_vllm.layers.attention import Attention
from mini_vllm.layers.attn_metadata import AttentionMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")

H, HKV, D, BS, T = 12, 2, 128, 16, 64
SCALE = D ** -0.5
TOL = 1e-2                 # 正常情况相对误差约 1.5e-3；出 bug 时是 1.0
# 修复后真实数据（第 0 层）缩放后分数最小约 -5400、最大约 2.4 万，-20000 已留出余量
CASES = [-10, -5000, -20000]
# 已知限制：缩放后分数仍低于阈值时照样会坏。strict=True：哪天上游修了，这条会变成 XPASS 报出来。
# 实测（2026-10-16，sm_89）：prefill 在 -5e4 附近坏；decode 更早，-40000 就坏（约 -5e4/log2e = -34700，
# 推测 decode 路径是在乘过 log2e 的分数上和 -5e4 比）。真实数据最低约 -5400，余量 decode 约 6x、prefill 约 9x。
KNOWN_LIMIT = -60000
KNOWN_LIMIT_DECODE = -40000


@pytest.fixture(scope="module")
def backend():
    from mini_vllm.layers.flashinfer_backend import FlashInferBackend
    return FlashInferBackend(H, HKV, D, BS, dtype=torch.bfloat16, device=torch.device("cuda"))


def synth(s: float):
    """所有 q 相同、所有 k 相同，使每个分数恰好为 s，softmax 是均匀分布。"""
    a = 320.0
    q = torch.zeros(T, H, D, device="cuda"); q[..., 0] = s / (a * SCALE)
    k = torch.zeros(T, HKV, D, device="cuda"); k[..., 0] = a
    v = torch.randn(T, HKV, D, device="cuda")
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def gold(q, k, v, causal: bool):
    o = F.scaled_dot_product_attention(q.float().transpose(0, 1)[None], k.float().transpose(0, 1)[None],
                                       v.float().transpose(0, 1)[None], is_causal=causal, enable_gqa=True)
    return o[0].transpose(0, 1)


def rel(o, g) -> float:
    return ((o.float() - g).norm() / g.norm()).item()


def layer(k_cache=None, v_cache=None):
    """Attention 的这两个方法只用到这几个属性，不必建整层。"""
    return types.SimpleNamespace(fi_q_scale=SCALE, k_cache=k_cache, v_cache=v_cache,
                                 num_kv_heads=HKV, head_dim=D)


def prefill_err(backend, s) -> float:
    q, k, v = synth(s)
    md = AttentionMetadata(is_prefill=True, slot_mapping=torch.zeros(T, dtype=torch.int32, device="cuda"),
                           wrapper=backend.plan_prefill(torch.tensor([0, T], dtype=torch.int32)))
    return rel(Attention._prefill_attn(layer(), q, k, v, md), gold(q, k, v, causal=True))


def decode_err(backend, s) -> float:
    q, k, v = synth(s)
    n = T // BS
    md = AttentionMetadata(is_prefill=False, slot_mapping=torch.zeros(1, dtype=torch.int32, device="cuda"),
                           wrapper=backend.plan_decode(torch.tensor([0, n], dtype=torch.int32),
                                                       torch.arange(n, dtype=torch.int32),
                                                       torch.tensor([BS], dtype=torch.int32)))
    kc, vc = k.view(n, BS, HKV, D).contiguous(), v.view(n, BS, HKV, D).contiguous()
    out = Attention._decode_attn(layer(kc, vc), q[-1:].contiguous(), md)
    return rel(out, gold(q, k, v, causal=False)[-1:])      # 最后一个 query 看全部 key


@pytest.mark.parametrize("s", CASES)
def test_prefill_negative_scores(backend, s):
    assert prefill_err(backend, s) < TOL


@pytest.mark.parametrize("s", CASES)
def test_decode_negative_scores(backend, s):
    assert decode_err(backend, s) < TOL


@pytest.mark.xfail(strict=True, reason="flashinfer math::inf = 5e4，缩放后分数 < -5e4 仍整行输出 0")
def test_known_limit(backend):
    assert prefill_err(backend, KNOWN_LIMIT) < TOL


@pytest.mark.xfail(strict=True, reason="decode 阈值约为缩放后分数 -34700，比 prefill 更低")
def test_known_limit_decode(backend):
    assert decode_err(backend, KNOWN_LIMIT_DECODE) < TOL
