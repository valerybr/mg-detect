"""Building blocks: ResNet generator, PatchGAN discriminator, PatchMLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(channels: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)


def init_weights(net: nn.Module) -> nn.Module:
    """Xavier normal initialization (gain=0.02), matching original CUT."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net


def _get_filter(filt_size: int) -> torch.Tensor:
    """Return a 2-D blur kernel (outer product of 1-D binomial filter)."""
    kernels = {
        1: [1.0],
        2: [1.0, 1.0],
        3: [1.0, 2.0, 1.0],
        4: [1.0, 3.0, 3.0, 1.0],
        5: [1.0, 4.0, 6.0, 4.0, 1.0],
    }
    a = torch.tensor(kernels[filt_size])
    filt = a[:, None] * a[None, :]
    return filt / filt.sum()


class Downsample(nn.Module):
    """Anti-aliased downsampling (Zhang, ICML 2019).

    Applies a fixed blur kernel followed by strided subsampling.
    Used as ``Conv(stride=1) → Downsample(stride=2)`` to replace
    ``Conv(stride=2)``, reducing aliasing artifacts.
    """

    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2):
        super().__init__()
        self.stride = stride
        pad_size = (filt_size - 1) // 2
        self.pad = nn.ReflectionPad2d([pad_size] * 4) # type: ignore
        filt = _get_filter(filt_size)
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.pad(x), self.filt, stride=self.stride, groups=x.shape[1]) # type: ignore 


class Upsample(nn.Module):
    """Anti-aliased upsampling (Zhang, ICML 2019).

    Applies transposed convolution with a fixed blur kernel for upsampling.
    Used as ``Upsample(stride=2) → Conv(stride=1)`` to replace
    ``ConvTranspose2d(stride=2)``.
    """

    def __init__(self, channels: int, filt_size: int = 4, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.filt_odd = filt_size % 2 == 1
        pad_size = (filt_size - 1) // 2
        self.pad = nn.ReplicationPad2d([1, 1, 1, 1]) #type: ignore
        filt = _get_filter(filt_size) * (stride ** 2)
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.conv_pad = 1 + pad_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv_transpose2d(
            self.pad(x), self.filt, #type: ignore - defined using register_buffer
            stride=self.stride, padding=self.conv_pad, groups=x.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return out
        return out[:, :, :-1, :-1]


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            _norm(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            _norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ResnetGenerator(nn.Module):
    """ResNet-based generator with anti-aliased down/upsampling.

    All layers are stored in a single flat ``nn.ModuleList`` so that any layer
    can be addressed by a global index, matching the original CUT paper's
    ``nce_layers`` convention (default ``[0, 4, 8, 12, 16]``).

    Layer index map (n_blocks=9, ngf=64):
         0  ReflectionPad2d(3)
         1  Conv2d(in_ch → ngf, 7×7)
         2  InstanceNorm2d
         3  ReLU
         4  Conv2d(ngf → ngf×2, 3×3, stride=1, pad=1)
         5  InstanceNorm2d
         6  ReLU
         7  Downsample(ngf×2)
         8  Conv2d(ngf×2 → ngf×4, 3×3, stride=1, pad=1)
         9  InstanceNorm2d
        10  ReLU
        11  Downsample(ngf×4)
        12  ResBlock #1  ┐
        …                │ n_blocks entries
        20  ResBlock #9  ┘
        21  Upsample(ngf×4)
        22  Conv2d(ngf×4 → ngf×2, 3×3, stride=1, pad=1)
        23  InstanceNorm2d
        24  ReLU
        25  Upsample(ngf×2)
        26  Conv2d(ngf×2 → ngf, 3×3, stride=1, pad=1)
        27  InstanceNorm2d
        28  ReLU
        29  ReflectionPad2d(3)
        30  Conv2d(ngf → out_ch, 7×7)
        31  Tanh
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 ngf: int = 64, n_blocks: int = 9):
        super().__init__()

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            _norm(ngf),
            nn.ReLU(inplace=True),
        ]
        c = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, 3, stride=1, padding=1),
                _norm(c * 2),
                nn.ReLU(inplace=True),
                Downsample(c * 2),
            ]
            c *= 2
        for _ in range(n_blocks):
            layers.append(_ResBlock(c))
        for _ in range(2):
            layers += [
                Upsample(c),
                nn.Conv2d(c, c // 2, 3, stride=1, padding=1),
                _norm(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return x

    def forward_with_features(
        self, x: torch.Tensor, nce_layers: list[int]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Full forward pass; collect intermediate outputs at ``nce_layers`` indices."""
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in nce_layers:
                feats.append(x)
        return x, feats


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with anti-aliased downsampling.

    Uses ``Conv(4×4, stride=1) → Downsample`` instead of ``Conv(stride=2)``
    to match the original CUT repo.  LeakyReLU(0.2), InstanceNorm (skipped
    on first layer).
    """

    def __init__(self, in_channels: int = 1, ndf: int = 64):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, 4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Downsample(ndf),
        ]
        c = ndf
        for i in range(1, 4):
            c_next = min(c * 2, ndf * 8)
            layers.append(nn.Conv2d(c, c_next, 4, stride=1, padding=1))
            layers.append(_norm(c_next))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i < 3:  # downsample for first 2 intermediate blocks
                layers.append(Downsample(c_next))
            c = c_next
        layers.append(nn.Conv2d(c, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PatchMLP  (one per NCE layer)
# ---------------------------------------------------------------------------

class PatchMLP(nn.Module):
    """Two-layer MLP that projects encoder features to a 256-dim NCE space.

    Operates on flattened spatial patches [N, C_in] → [N, 256] (L2-normed).
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C]
        x = self.net(x)
        return F.normalize(x, dim=1)
