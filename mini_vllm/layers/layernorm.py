import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * (torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        output = self._norm(x)
        return output.to(input_dtype) * self.weight

if __name__ == "__main__":
    torch.manual_seed(0)
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8)
    y = norm(x)

    assert y.shape == (2, 3, 8)

    assert torch.allclose(y.pow(2).mean(dim=-1).sqrt(), torch.ones(2, 3), atol=1e-8)

    assert torch.allclose(y, nn.RMSNorm(8, eps=1e-6)(x), atol=1e-8)

    y.sum().backward()
    assert norm.weight.grad is not None

    print("RMSNorm pass!")