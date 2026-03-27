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

    Architecture:
        Encoder  : ReflectionPad → Conv(1→ngf) → 2× stride-2 downsampling
        Bottleneck: n_blocks residual blocks
        Decoder  : 2× upsampling → ReflectionPad → Conv(ngf→out_channels) → Tanh

    Intermediate encoder activations are exposed via ``encode`` so that
    PatchNCE can extract multi-scale features.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 ngf: int = 64, n_blocks: int = 9):
        super().__init__()

        # Build encoder layers individually so we can index into them.
        enc: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            _norm(ngf),
            nn.ReLU(inplace=True),
        ]
        c = ngf
        for _ in range(2):
            enc += [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                _norm(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        self.encoder = nn.ModuleList(enc)

        self.bottleneck = nn.Sequential(*[_ResBlock(c) for _ in range(n_blocks)])

        dec: list[nn.Module] = []
        for _ in range(2):
            dec += [
                nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1,
                                   output_padding=1),
                _norm(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2
        dec += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_channels, 7),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*dec)

    # ------------------------------------------------------------------
    # nce_layers refers to indices into self.encoder (0-based).
    # Typical choice: [0, 4, 8] — after first conv, after 1st downsample,
    # after 2nd downsample.
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor,
               nce_layers: list[int]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run encoder, collect intermediate features at requested layer indices.

        Returns:
            feat_last : final encoder output (input to bottleneck)
            feats     : list of intermediate tensors at nce_layers indices
        """
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in nce_layers:
                feats.append(x)
        return x, feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        x = self.bottleneck(x)
        return self.decoder(x)

    def forward_with_features(
        self, x: torch.Tensor, nce_layers: list[int]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Full forward pass + intermediate encoder features for PatchNCE."""
        enc_out, feats = self.encode(x, nce_layers)
        out = self.decoder(self.bottleneck(enc_out))
        return out, feats


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
