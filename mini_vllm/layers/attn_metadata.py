from dataclasses import dataclass
import torch
from typing import Optional,Any

@dataclass
class AttentionMetadata:
    is_prefill: bool
    slot_mapping: torch.Tensor
    # prefill 用
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: int = 0
    # decode 用
    seq_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    wrapper: Optional[Any] = None
    has_cached: bool = False