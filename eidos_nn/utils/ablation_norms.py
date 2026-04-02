import torch
import torch.nn as nn


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RMSNorm1d(nn.Module):
    """
    Minimal RMS normalization over the last dimension.
    This preserves per-sample independence and avoids mean subtraction.
    """

    def __init__(self, dim: int, eps: float = 1e-6, learnable_scale: bool = True):
        super().__init__()
        self.eps = eps
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("scale", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x32 = x.float()
        rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + self.eps)
        out = (x32 / rms) * self.scale.float()
        return out.to(x_dtype)


class RMSNorm2d(nn.Module):
    """
    Per-sample, per-channel RMS normalization over spatial dimensions.
    """

    def __init__(self, num_features: int, eps: float = 1e-6, learnable_scale: bool = True):
        super().__init__()
        self.eps = eps
        if learnable_scale:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer("weight", torch.ones(num_features))
            self.register_buffer("bias", torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x32 = x.float()
        rms = torch.sqrt(torch.mean(x32 * x32, dim=tuple(range(2, x.dim())), keepdim=True) + self.eps)
        out = x32 / rms
        shape = [1, x.shape[1]] + [1] * (x.dim() - 2)
        out = out * self.weight.float().view(*shape) + self.bias.float().view(*shape)
        return out.to(x_dtype)
