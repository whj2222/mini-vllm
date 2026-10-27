import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, base: float):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)  # (D/2,)
        t = torch.arange(max_position, dtype=torch.float) # (max_position,)
        freqs = torch.outer(t, inv_freq) # (max_position, D/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (max_position, D)
        cos = emb.cos()
        sin = emb.sin()

        self.register_buffer('cos_cache', cos, persistent=False)
        self.register_buffer('sin_cache', sin, persistent=False)

    def forward(self, positions: torch.Tensor, q:torch.Tensor, k:torch.Tensor):
        cos = self.cos_cache[positions].to(q.dtype).unsqueeze(1)
        sin = self.sin_cache[positions].to(q.dtype).unsqueeze(1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k