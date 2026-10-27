import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mini_vllm.layers.rotary import RotaryEmbedding
from mini_vllm.layers.attn_metadata import AttentionMetadata
from mini_vllm.kernel.store_kv import store_kv

class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rotary: RotaryEmbedding):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim # 1536
        self.kv_size = num_kv_heads * head_dim # 256
        self.k_cache = None
        self.v_cache = None
        # flashinfer 0.6.18 的 "-inf" 是 -5e4（math.cuh:33），running max 按未缩放的 q·k 比较；
        # 一行合法分数全低于 -5e4 时整行输出 0。Qwen2.5 第 0 层 |k|≈320，原始分数能到 -6 万。
        # 所以 q 先乘 1/sqrt(D) 再交给 flashinfer（plan 里 sm_scale=1.0），分数整体缩小约 11 倍。
        # torch 分支不用：SDPA 自己在 fp32 分数上乘 scale。
        self.fi_q_scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=True)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        self.rotary = rotary

    def _store_kv_torch(self, k: Tensor, v: Tensor, slot_mapping) -> None:
        if self.k_cache is None:
            return
        slots = slot_mapping.long()
        self.k_cache.view(-1, self.num_kv_heads, self.head_dim)[slots] = k
        self.v_cache.view(-1, self.num_kv_heads, self.head_dim)[slots] = v

    def _prefill_attn(self, q: Tensor, k: Tensor, v: Tensor, md: AttentionMetadata) -> Tensor:
        if md.wrapper is not None:
            if md.has_cached:
                return md.wrapper.run(q * self.fi_q_scale, (self.k_cache, self.v_cache))
            else:
                return md.wrapper.run(q * self.fi_q_scale, k, v)
        else:
            if md.has_cached:
                cu = md.cu_seqlens.tolist()
                seq_lens = md.seq_lens.tolist()
                bs = self.k_cache.shape[1]
                outs = []
                for b, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
                    Lq = e - s
                    L = seq_lens[b]
                    n = (L + bs - 1) // bs
                    blocks = md.block_tables[b, :n].long()
                    kk = self.k_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:L]
                    vv = self.v_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:L]
                    row = torch.arange(Lq, device=q.device)[:, None]
                    col = torch.arange(L, device=q.device)[None, :]
                    mask = col <= row + (L - Lq)
                    o = F.scaled_dot_product_attention(q[s:e].transpose(0, 1).unsqueeze(0),
                                                       kk.transpose(0, 1).unsqueeze(0),
                                                       vv.transpose(0, 1).unsqueeze(0),
                                                       attn_mask=mask, enable_gqa=True)
                    outs.append(o.squeeze(0).transpose(0, 1))  # [Lq, H, D]
                return torch.cat(outs, dim=0)
            else:
                cu = md.cu_seqlens.tolist()
                outs = []
                for s, e in zip(cu[:-1], cu[1:]):
                    o = F.scaled_dot_product_attention(q[s:e].transpose(0, 1).unsqueeze(0),
                                                       k[s:e].transpose(0, 1).unsqueeze(0),
                                                       v[s:e].transpose(0, 1).unsqueeze(0),
                                                       is_causal=True, enable_gqa=True)
                    outs.append(o.squeeze(0).transpose(0, 1))
                return torch.cat(outs, dim=0)

    def _decode_attn(self, q: Tensor, md: AttentionMetadata) -> Tensor:
        if md.wrapper is not None:
            return md.wrapper.run(q * self.fi_q_scale, (self.k_cache, self.v_cache))
        else:
            # q: [B, H, D]，每条 seq 一个 query；KV 从池子按 block_table 取回
            bs = self.k_cache.shape[1]
            outs = []
            for b, L in enumerate(md.seq_lens.tolist()):
                n = (L + bs - 1) // bs
                blocks = md.block_tables[b, :n].long()     # 只取有效块，补齐的 -1 碰不到
                k = self.k_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:L]  # [L, Hkv, D]
                v = self.v_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:L]
                o = F.scaled_dot_product_attention(
                    q[b:b + 1].transpose(0, 1).unsqueeze(0),   # [1, H,   1, D]
                    k.transpose(0, 1).unsqueeze(0),            # [1, Hkv, L, D]
                    v.transpose(0, 1).unsqueeze(0),
                    enable_gqa=True,                           # 唯一的 query 在最后，能看到全部，不需要 mask
                )
                outs.append(o.squeeze(0).transpose(0, 1))      # [1, H, D]
            return torch.cat(outs, dim=0)                      # [B, H, D]



    def forward(self, positions: Tensor, x: Tensor, attn_metadata: AttentionMetadata) -> Tensor:
        # positions: [T]  x: [T, hidden]
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.num_heads, self.head_dim)     # [T, 12, 128]
        k = self.k_proj(x).view(T, self.num_kv_heads, self.head_dim)  # [T, 2, 128]
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(positions, q, k)

        if self.k_cache is not None:
            store_kv(k, v, self.k_cache, self.v_cache, attn_metadata.slot_mapping)

        if attn_metadata.is_prefill:
            out = self._prefill_attn(q, k, v, attn_metadata)   # [T, H, D]
        else:
            out = self._decode_attn(q, attn_metadata)          # [B, H, D]，decode 时 T == B
        return self.o_proj(out.reshape(T, self.q_size))

