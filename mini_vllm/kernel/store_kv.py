import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _store_kv_kernel(k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, k_stride, v_stride, slot_ptr, KV_DIM: tl.constexpr):
    i = tl.program_id(0)
    offset = tl.arange(0, KV_DIM)
    slot = tl.load(slot_ptr + i)
    if slot < 0:
        return
    k = tl.load(k_ptr + i * k_stride + offset)
    v = tl.load(v_ptr + i * v_stride + offset)
    slot = slot.to(tl.int64)
    tl.store(k_cache_ptr + slot * KV_DIM + offset, k)
    tl.store(v_cache_ptr + slot * KV_DIM + offset, v)

def store_kv(k: Tensor, v: Tensor, k_cache: Tensor, v_cache: Tensor, slot_mapping: Tensor) -> None:
    T = k.shape[0]
    Hkv = k.shape[1]
    D = k.shape[2]
    assert v.shape == k.shape
    assert k.stride(1) == D and k.stride(2) == 1 and v.stride(1) == D and v.stride(2) == 1
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    if T == 0:
        return
    KV_DIM = Hkv * D
    k_stride = k.stride(0)
    v_stride = v.stride(0)
    _store_kv_kernel[(T,)](k, v, k_cache, v_cache, k_stride, v_stride, slot_mapping, KV_DIM)