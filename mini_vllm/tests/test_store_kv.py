"""Triton store_kv 单测：和 Attention._store_kv_torch 写出的池子逐位比较（torch.equal，不允许误差）。

端到端测试只能间接证明 store_kv 对，下面这些情况它覆盖不到：
slot = -1 跳过（CUDA Graph padding 要用）、源张量第 0 维不连续（以后合并 qkv）、
T = 0、池子最后一个 slot、按层切出的池子视图、违反假设时断言能拦住。

运行（项目根目录）：python -m pytest mini_vllm/tests/test_store_kv.py -v
"""
import types

import pytest
import torch

from mini_vllm.kernel.store_kv import store_kv
from mini_vllm.layers.attention import Attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")

HKV, D = 2, 128
NUM_SLOTS = 2048                       # 池子总 slot 数，num_blocks = NUM_SLOTS // block_size
DT = torch.bfloat16


def make_pool(block_size: int, layers: int | None = None) -> torch.Tensor:
    """随机初始化（不是 0），这样"没被写到的位置保持不变"才有意义。"""
    shape = (NUM_SLOTS // block_size, block_size, HKV, D)
    if layers is not None:
        shape = (2, layers) + shape    # 和 ModelRunner._allocate_kv_cache 同布局
    return torch.randn(shape, dtype=DT, device="cuda")


def make_kv(T: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.randn(T, HKV, D, dtype=DT, device="cuda"),
            torch.randn(T, HKV, D, dtype=DT, device="cuda"))


def torch_store(k_cache, v_cache, k, v, slots) -> None:
    """调用 Attention._store_kv_torch 当参考。它只用到下面四个属性，不必建整层。"""
    fake = types.SimpleNamespace(k_cache=k_cache, v_cache=v_cache, num_kv_heads=HKV, head_dim=D)
    Attention._store_kv_torch(fake, k, v, slots)


def random_slots(T: int) -> torch.Tensor:
    return torch.randperm(NUM_SLOTS, device="cuda")[:T].to(torch.int32)


@pytest.mark.parametrize("block_size", [16, 1])
@pytest.mark.parametrize("T", [1, 7, 300])
def test_matches_torch(block_size, T):
    k, v = make_kv(T)
    slots = random_slots(T)
    kc, vc = make_pool(block_size), make_pool(block_size)
    kc_ref, vc_ref = kc.clone(), vc.clone()

    store_kv(k, v, kc, vc, slots)
    torch_store(kc_ref, vc_ref, k, v, slots)
    assert torch.equal(kc, kc_ref) and torch.equal(vc, vc_ref)


def test_negative_slots_skipped():
    # 注意不能拿 _store_kv_torch 当参考：torch 下标 -1 会写到最后一行
    T = 64
    k, v = make_kv(T)
    slots = random_slots(T)
    slots[::3] = -1
    kc, vc = make_pool(16), make_pool(16)
    expect_k, expect_v = kc.clone(), vc.clone()
    ok = slots >= 0
    expect_k.view(-1, HKV, D)[slots[ok].long()] = k[ok]
    expect_v.view(-1, HKV, D)[slots[ok].long()] = v[ok]

    store_kv(k, v, kc, vc, slots)
    assert torch.equal(kc, expect_k) and torch.equal(vc, expect_v)


def test_noncontiguous_source():
    # 模拟合并 qkv：k/v 是从 [T, q+k+v] 里切出来的视图，stride(0) = 2048 != Hkv*D
    T = 100
    qkv = torch.randn(T, 1536 + 2 * HKV * D, dtype=DT, device="cuda")
    k = qkv[:, 1536:1536 + HKV * D].view(T, HKV, D)
    v = qkv[:, 1536 + HKV * D:].view(T, HKV, D)
    assert not k.is_contiguous() and k.stride() == (2048, D, 1)
    slots = random_slots(T)
    kc, vc = make_pool(16), make_pool(16)
    kc_ref, vc_ref = kc.clone(), vc.clone()

    store_kv(k, v, kc, vc, slots)
    torch_store(kc_ref, vc_ref, k.contiguous(), v.contiguous(), slots)
    assert torch.equal(kc, kc_ref) and torch.equal(vc, vc_ref)


def test_empty():
    k, v = make_kv(0)
    kc, vc = make_pool(16), make_pool(16)
    before = kc.clone()
    store_kv(k, v, kc, vc, torch.empty(0, dtype=torch.int32, device="cuda"))
    assert torch.equal(kc, before)


def test_last_slot():
    k, v = make_kv(2)
    slots = torch.tensor([NUM_SLOTS - 1, 0], dtype=torch.int32, device="cuda")
    kc, vc = make_pool(16), make_pool(16)
    store_kv(k, v, kc, vc, slots)
    assert torch.equal(kc.view(-1, HKV, D)[-1], k[0]) and torch.equal(kc.view(-1, HKV, D)[0], k[1])
    assert torch.equal(vc.view(-1, HKV, D)[-1], v[0]) and torch.equal(vc.view(-1, HKV, D)[0], v[1])


def test_layer_view():
    # ModelRunner 把 [2, L, N, bs, Hkv, D] 按层切成视图挂到每层；写第 3 层不能碰到别的层
    L, layer, T = 4, 3, 50
    pool = make_pool(16, layers=L)
    before = pool.clone()
    k, v = make_kv(T)
    slots = random_slots(T)

    store_kv(k, v, pool[0, layer], pool[1, layer], slots)
    torch_store(before[0, layer], before[1, layer], k, v, slots)
    assert torch.equal(pool, before)


def test_bad_layout_rejected():
    # 头维度不连续（stride(1) != D）时 kernel 的地址算法不成立，必须被断言拦住
    T = 8
    k = torch.randn(T, D, HKV, dtype=DT, device="cuda").transpose(1, 2)
    v = k.clone()
    kc, vc = make_pool(16), make_pool(16)
    with pytest.raises(AssertionError):
        store_kv(k, v, kc, vc, random_slots(T))
