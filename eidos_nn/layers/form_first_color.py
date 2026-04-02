import torch
import torch.nn as nn
import torch.nn.functional as F

from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm


class FormFirstColorLayer(nn.Module):
    """
    Form-First color front-end.

    Discretizes raw RGB into ternary states (-1, 0, +1) and optionally appends
    CMYK channels as a second color basis. No smoothing or averaging is used.
    """

    def __init__(
        self,
        use_cmyk: bool = False,
        color_depth: int = 4,
        adaptive_depth: bool = True,
        depth_patch_size: int = 4,
        depth_sensitivity: float = 0.05,
        use_gridnorm: bool = True,
        ternary_threshold: float = 0.1,
        assume_normalized: bool = True,
        rgb_mean=(0.4914, 0.4822, 0.4465),
        rgb_std=(0.2023, 0.1994, 0.2010),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.use_cmyk = use_cmyk
        self.color_depth = max(1, int(color_depth))
        self.adaptive_depth = adaptive_depth
        self.depth_patch_size = max(1, int(depth_patch_size))
        self.depth_sensitivity = depth_sensitivity
        self.use_gridnorm = use_gridnorm
        self.ternary_threshold = ternary_threshold
        self.assume_normalized = assume_normalized
        self.eps = eps
        self.out_channels = 3 + (4 if use_cmyk else 0)

        mean = torch.tensor(rgb_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(rgb_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", mean)
        self.register_buffer("rgb_std", std)

        # Optional FP64 geometric normalization for color vectors
        self.rgb_norm = ModularPhaseNorm(3, base=7) if use_gridnorm else None
        self.cmyk_norm = ModularPhaseNorm(4, base=7) if use_gridnorm else None

    def _to_raw_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if self.assume_normalized:
            x = x * self.rgb_std + self.rgb_mean
        return x.clamp(0.0, 1.0)

    def _apply_gridnorm(self, x: torch.Tensor, norm: ModularPhaseNorm) -> torch.Tensor:
        if norm is None:
            return x
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x_norm = norm(x_flat).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_norm

    def _ternary(self, x: torch.Tensor) -> torch.Tensor:
        mid = 0.5
        delta = self.ternary_threshold
        pos = x > (mid + delta)
        neg = x < (mid - delta)
        return torch.where(pos, torch.ones_like(x), torch.where(neg, -torch.ones_like(x), torch.zeros_like(x)))

    def _quantize(self, x: torch.Tensor, depth_map: torch.Tensor = None) -> torch.Tensor:
        if self.color_depth == 1:
            return self._ternary(x)

        if depth_map is not None:
            levels = 2.0 * depth_map + 1.0
            step = 2.0 / (levels - 1.0)
            x_scaled = x * 2.0 - 1.0
            q = torch.round((x_scaled + 1.0) / step) * step - 1.0
            return q

        levels = 2 * self.color_depth + 1
        # Map [0,1] -> [-1,1] then quantize to uniform levels.
        x_scaled = x * 2.0 - 1.0
        step = 2.0 / (levels - 1)
        q = torch.round((x_scaled + 1.0) / step) * step - 1.0
        return q

    def _compute_depth_map(self, rgb_raw: torch.Tensor) -> torch.Tensor:
        # Luminance proxy for structure: no smoothing, only measurement.
        r = rgb_raw[:, 0:1]
        g = rgb_raw[:, 1:2]
        b = rgb_raw[:, 2:3]
        lum = 0.2989 * r + 0.5870 * g + 0.1140 * b

        k = self.depth_patch_size
        mean = F.avg_pool2d(lum, kernel_size=k, stride=k, ceil_mode=True)
        mean2 = F.avg_pool2d(lum * lum, kernel_size=k, stride=k, ceil_mode=True)
        var = (mean2 - mean * mean).clamp(min=0.0)

        # Normalize variance into [0,1) with a soft saturation.
        var_norm = var / (var + self.depth_sensitivity)
        depth = 1.0 + torch.round((self.color_depth - 1) * var_norm)
        depth = depth.clamp(1.0, float(self.color_depth))

        # Upsample to full resolution for per-pixel quantization.
        depth_map = F.interpolate(depth, size=rgb_raw.shape[-2:], mode='nearest')
        return depth_map

    def _rgb_to_cmyk(self, rgb: torch.Tensor) -> torch.Tensor:
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]
        k = 1.0 - torch.max(rgb, dim=1, keepdim=True).values
        denom = 1.0 - k + self.eps
        c = (1.0 - r - k) / denom
        m = (1.0 - g - k) / denom
        y = (1.0 - b - k) / denom
        cmyk = torch.cat([c, m, y, k], dim=1)
        return cmyk.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"FormFirstColorLayer expects 3-channel RGB input, got {x.shape[1]} channels.")

        rgb_raw = self._to_raw_rgb(x)
        depth_map = None
        if self.adaptive_depth and self.color_depth > 1:
            depth_map = self._compute_depth_map(rgb_raw)

        if self.use_gridnorm and self.rgb_norm is not None:
            rgb_raw = self._apply_gridnorm(rgb_raw, self.rgb_norm)

        rgb_q = self._quantize(rgb_raw, depth_map=depth_map)

        if not self.use_cmyk:
            return rgb_q

        cmyk_raw = self._rgb_to_cmyk(rgb_raw)
        if self.use_gridnorm and self.cmyk_norm is not None:
            cmyk_raw = self._apply_gridnorm(cmyk_raw, self.cmyk_norm)

        cmyk_q = self._quantize(cmyk_raw, depth_map=depth_map)
        return torch.cat([rgb_q, cmyk_q], dim=1)
