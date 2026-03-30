"""Building blocks: ResNet generator, PatchGAN discriminator, PatchMLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(channels: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)


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
    """ResNet-based generator for unpaired image-to-image translation.

    All layers are stored in a single flat ``nn.ModuleList`` so that any layer
    can be addressed by a global index, matching the original CUT paper's
    ``nce_layers`` convention (default ``[0, 4, 8, 12, 16]``).

    Layer index map (n_blocks=9, ngf=64):
        0  ReflectionPad2d(3)
        1  Conv2d(in_ch → ngf, 7×7)
        2  InstanceNorm2d
        3  ReLU
        4  Conv2d(ngf → ngf×2, stride=2)
        5  InstanceNorm2d
        6  ReLU
        7  Conv2d(ngf×2 → ngf×4, stride=2)
        8  InstanceNorm2d
        9  ReLU
        10 ResBlock #1  ┐
        …               │ n_blocks entries
        18 ResBlock #9  ┘
        19 ConvTranspose2d(ngf×4 → ngf×2)
        20 InstanceNorm2d
        21 ReLU
        22 ConvTranspose2d(ngf×2 → ngf)
        23 InstanceNorm2d
        24 ReLU
        25 ReflectionPad2d(3)
        26 Conv2d(ngf → out_ch, 7×7)
        27 Tanh
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
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                _norm(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2
        for _ in range(n_blocks):
            layers.append(_ResBlock(c))
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1,
                                   output_padding=1),
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
    """70×70 PatchGAN discriminator (5 conv layers, stride-2 downsampling).

    Output is a 32×32 real/fake map; each unit has a ~70-pixel receptive field.
    Uses LeakyReLU(0.2) and InstanceNorm (skipped on first layer).
    """

    def __init__(self, in_channels: int = 1, ndf: int = 64):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        c = ndf
        for i in range(1, 4):
            c_next = min(c * 2, ndf * 8)
            stride = 2 if i < 3 else 1
            layers += [
                nn.Conv2d(c, c_next, 4, stride=stride, padding=1),
                _norm(c_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
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
