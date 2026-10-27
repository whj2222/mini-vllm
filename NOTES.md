## 目前要做什么
* attention.py中_store_kv把本步的K/V按照slot写进池子。
  * torch高级索引每层要起好几个小kernel，所以要换成自己写的Triton kernel
* attention.py中_prefill_attn每条序列各自causal attention。
  * 换成flashinfer的varlen(ragged)接口
* attention.py中_decode_attn 按block_table把kv取出来再算
  * 改成flashinfer的paged接口

## Triton kernel
1. 新增/kernel/__init__.py
2. 新增/kernel/store_kv.py
3. 在store_kv.py写store_kv(k: Tensor, v: Tensor, k_cache: Tensor, v_kache: Tensor,
                           slot_mapping: Tensor) -> None
                  _store_kv_kernel(k_ptr, v_ptr, k_cache_ptr, k_stride, v_stride,
                           v_cache_ptr, slot_ptr, KV_DIM: tl.constexptr)

## flashinfer decode接口
