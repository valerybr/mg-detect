"""CycleCUT model: forward pass, optimizer step, and checkpointing."""

import itertools
from typing import Optional

import torch
import torch.nn as nn

from .loss import LSGANLoss, PatchNCELoss
from .networks import ResnetGenerator, PatchGANDiscriminator, PatchMLP

class CycleCUTModel(nn.Module):
    """CycleCUT: CycleGAN + PatchNCE (CUT) contrastive loss.

    Trains two generators (G_AB, G_BA) and two discriminators (D_A, D_B).
    PatchNCE is applied at ``nce_layers`` encoder stages of both generators.

    Args:
        device        : torch device
        ngf           : base channel count for generators (default 64)
        ndf           : base channel count for discriminators (default 64)
        n_blocks      : residual blocks in generator bottleneck (default 9)
        nce_layers    : encoder layer indices to use for PatchNCE (default [0,4,8])
        num_patches   : patches randomly sampled per NCE layer (default 256)
        temperature   : PatchNCE temperature τ (default 0.07)
        lambda_nce    : weight on PatchNCE loss (default 1.0)
        lambda_cyc    : weight on cycle-consistency L1 loss (default 10.0)
        lambda_idt    : weight on identity L1 loss (default 5.0)
        lr            : Adam learning rate (default 2e-4)
        beta1         : Adam β₁ (default 0.5)
        n_epochs      : epochs with constant LR (default 100)
        n_epochs_decay: epochs over which LR linearly decays to 0 (default 100)
    """

    def __init__(
        self,
        device: torch.device,
        ngf: int = 64,
        ndf: int = 64,
        n_blocks: int = 9,
        nce_layers: list[int] = [3, 6, 9],
        num_patches: int = 256,
        temperature: float = 0.07,
        lambda_nce: float = 1.0,
        lambda_cyc: float = 10.0,
        lambda_idt: float = 5.0,
        lr: float = 2e-4,
        beta1: float = 0.5,
        n_epochs: int = 100,
        n_epochs_decay: int = 100,
    ):
        super().__init__()
        self.device = device
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.n_blocks = n_blocks
        self.lambda_nce = lambda_nce
        self.lambda_cyc = lambda_cyc
        self.lambda_idt = lambda_idt
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay

        # ----- networks -----
        self.G_AB = ResnetGenerator(n_blocks=n_blocks).to(device)
        self.G_BA = ResnetGenerator(n_blocks=n_blocks).to(device)
        self.D_A = PatchGANDiscriminator(ndf=ndf).to(device)
        self.D_B = PatchGANDiscriminator(ndf=ndf).to(device)

        # One PatchMLP per NCE layer; channel widths come from the generator.
        # For ngf=64: layer 0 → 64ch, layer 4 → 128ch, layer 8 → 256ch
        nce_channels = self._nce_channel_sizes(ngf, n_blocks)
        self.mlps_AB = nn.ModuleList(
            [PatchMLP(c).to(device) for c in nce_channels]
        )
        self.mlps_BA = nn.ModuleList(
            [PatchMLP(c).to(device) for c in nce_channels]
        )

        # ----- losses -----
        self.crit_gan = LSGANLoss()
        self.crit_nce = [PatchNCELoss(temperature).to(device) for _ in nce_layers]
        self.crit_l1 = nn.L1Loss()

        # ----- optimizers -----
        g_params = itertools.chain(
            self.G_AB.parameters(), self.G_BA.parameters(),
            self.mlps_AB.parameters(), self.mlps_BA.parameters(),
        )
        self.opt_G = torch.optim.Adam(g_params, lr=lr, betas=(beta1, 0.999))
        self.opt_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=lr, betas=(beta1, 0.999),
        )

        # Linear decay: keep LR constant for n_epochs, then decay to 0
        total = n_epochs + n_epochs_decay
        def _lr_lambda(epoch: int) -> float:
            if epoch < n_epochs:
                return 1.0
            return max(0.0, 1.0 - (epoch - n_epochs) / n_epochs_decay)

        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, _lr_lambda)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, _lr_lambda)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nce_channel_sizes(self, ngf: int, n_blocks: int) -> list[int]:
        """Return output channel count at each nce_layer index.

        Flat generator layout (see ResnetGenerator docstring):
          0          ReflectionPad2d          → in_channels (1)
          1-3        Conv/Norm/ReLU           → ngf
          4-6        Conv(stride=2)/Norm/ReLU → ngf×2
          7-9        Conv(stride=2)/Norm/ReLU → ngf×4
          10..9+n    ResBlock ×n_blocks       → ngf×4
          ...        decoder layers
        """
        ch: dict[int, int] = {}
        ch[0] = 1
        for i in range(1, 4):
            ch[i] = ngf
        for i in range(4, 7):
            ch[i] = ngf * 2
        for i in range(7, 10):
            ch[i] = ngf * 4
        for i in range(10, 10 + n_blocks):
            ch[i] = ngf * 4
        return [ch[i] for i in self.nce_layers]

    @staticmethod
    def _sample_patches(
        feats: list[torch.Tensor], num_patches: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Flatten spatial dims and randomly sample up to num_patches positions.

        Each layer is sampled independently because they have different spatial
        resolutions (512×512, 256×256, 128×128 for nce_layers=[3,6,9]).

        Returns:
            sampled_feats : list of [n, C] tensors  (n ≤ num_patches)
            patch_ids     : list of [n] index tensors, one per layer
        """
        out: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        for feat in feats:
            B, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
            N = flat.shape[0]
            n = min(num_patches, N)
            idx = torch.randperm(N, device=feat.device)[:n]
            ids.append(idx)
            out.append(flat[idx])
        return out, ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def set_input(self, real_A: torch.Tensor, real_B: torch.Tensor):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        self.fake_B, self.feats_A = self.G_AB(self.real_A, self.nce_layers)
        self.fake_A, self.feats_B = self.G_BA(self.real_B, self.nce_layers)
        self.rec_A = self.G_BA(self.fake_B)
        self.rec_B = self.G_AB(self.fake_A)
        self.idt_A = self.G_BA(self.real_A)   # G_BA(A) ≈ A
        self.idt_B = self.G_AB(self.real_B)   # G_AB(B) ≈ B

    # ------------------------------------------------------------------
    # NCE loss
    # ------------------------------------------------------------------

    def _compute_nce_loss(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        G: ResnetGenerator,
        mlps: nn.ModuleList,
    ) -> torch.Tensor:
        """Compute mean PatchNCE loss across nce_layers.

        src : source image (real)
        tgt : generated image (fake, translated from src)
        """
        _, feats_src = G(src, self.nce_layers)
        _, feats_tgt = G(tgt, self.nce_layers)

        # Sample the same spatial positions in src and tgt per layer so that
        # query (tgt) and key (src) are spatially aligned.
        sampled_src, patch_ids = self._sample_patches(feats_src, self.num_patches)
        sampled_tgt: list[torch.Tensor] = []
        for feat, idx in zip(feats_tgt, patch_ids):
            B, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
            sampled_tgt.append(flat[idx])

        total = torch.tensor(0.0, device=self.device)
        for feat_src, feat_tgt, mlp, crit in zip(
            sampled_src, sampled_tgt, mlps, self.crit_nce
        ):
            q = mlp(feat_tgt)   # query: from generated image
            k = mlp(feat_src)   # key  : from source image (same positions)
            total = total + crit(q, k)
        return total / len(self.nce_layers)

    # ------------------------------------------------------------------
    # Discriminator step
    # ------------------------------------------------------------------

    def _update_D(self):
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad_(True)

        self.opt_D.zero_grad()

        loss_D_A = (
            self.crit_gan(self.D_A(self.real_A), True)
            + self.crit_gan(self.D_A(self.fake_A.detach()), False)
        ) * 0.5

        loss_D_B = (
            self.crit_gan(self.D_B(self.real_B), True)
            + self.crit_gan(self.D_B(self.fake_B.detach()), False)
        ) * 0.5

        (loss_D_A + loss_D_B).backward()
        self.opt_D.step()
        return loss_D_A.item(), loss_D_B.item()

    # ------------------------------------------------------------------
    # Generator step
    # ------------------------------------------------------------------

    def _update_G(self):
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad_(False)

        self.opt_G.zero_grad()

        # Adversarial
        loss_adv = (
            self.crit_gan(self.D_B(self.fake_B), True)
            + self.crit_gan(self.D_A(self.fake_A), True)
        )

        # PatchNCE (applied in both translation directions)
        loss_nce = (
            self._compute_nce_loss(self.real_A, self.fake_B, self.G_AB, self.mlps_AB)
            + self._compute_nce_loss(self.real_B, self.fake_A, self.G_BA, self.mlps_BA)
        ) * self.lambda_nce

        # Cycle consistency
        loss_cyc = (
            self.crit_l1(self.rec_A, self.real_A)
            + self.crit_l1(self.rec_B, self.real_B)
        ) * self.lambda_cyc

        # Identity
        loss_idt = (
            self.crit_l1(self.idt_A, self.real_A)
            + self.crit_l1(self.idt_B, self.real_B)
        ) * self.lambda_idt

        loss_G = loss_adv + loss_nce + loss_cyc + loss_idt
        loss_G.backward()
        self.opt_G.step()

        return {
            "adv": loss_adv.item(),
            "nce": loss_nce.item(),
            "cyc": loss_cyc.item(),
            "idt": loss_idt.item(),
            "G":   loss_G.item(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> dict:
        """Run one full optimization step (forward + D update + G update).

        Returns a dict of scalar loss values for logging.
        """
        self.forward()
        loss_D_A, loss_D_B = self._update_D()
        losses_G = self._update_G()
        return {
            "D_A": loss_D_A,
            "D_B": loss_D_B,
            **losses_G,
        }

    def scheduler_step(self):
        """Call once per epoch to advance LR schedulers."""
        self.sched_G.step()
        self.sched_D.step()

    def save(self, path: str, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "G_AB": self.G_AB.state_dict(),
                "G_BA": self.G_BA.state_dict(),
                "D_A":  self.D_A.state_dict(),
                "D_B":  self.D_B.state_dict(),
                "mlps_AB": self.mlps_AB.state_dict(),
                "mlps_BA": self.mlps_BA.state_dict(),
                "opt_G": self.opt_G.state_dict(),
                "opt_D": self.opt_D.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> int:
        """Load checkpoint; returns the saved epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.G_AB.load_state_dict(ckpt["G_AB"])
        self.G_BA.load_state_dict(ckpt["G_BA"])
        self.D_A.load_state_dict(ckpt["D_A"])
        self.D_B.load_state_dict(ckpt["D_B"])
        self.mlps_AB.load_state_dict(ckpt["mlps_AB"])
        self.mlps_BA.load_state_dict(ckpt["mlps_BA"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        return ckpt["epoch"]
