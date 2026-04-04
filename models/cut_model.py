"""CUT model (Contrastive Unpaired Translation, Park et al. ECCV 2020).

One generator G: A→B, one discriminator D_B, PatchNCE loss + optional
identity PatchNCE loss. No cycle consistency, no second generator.
"""

import itertools

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import  autocast

from .networks import ResnetGenerator, PatchGANDiscriminator, PatchMLP, init_weights
from .cut import LSGANLoss, PatchNCELoss


class CUTModel(nn.Module):
    """CUT: Contrastive Unpaired Translation (Park et al., ECCV 2020).

    Trains a single generator G (A→B) with:
    - Adversarial loss (LSGAN): makes G(A) indistinguishable from real B
    - PatchNCE loss: patch-level mutual information between A and G(A)
    - Identity PatchNCE (optional): patch correspondence between B and G(B),
      encouraging content preservation when translating within domain B

    No cycle consistency loss, no second generator.

    AMP (mixed precision) is managed internally: GradScalers and autocast
    contexts live here so the training loop can just call optimize().

    Args:
        device        : torch device
        ngf           : base channel count for generator (default 64)
        ndf           : base channel count for discriminator (default 64)
        n_blocks      : residual blocks in generator bottleneck (default 9)
        nce_layers    : encoder layer indices for PatchNCE (default [0, 4, 8, 12, 16])
        num_patches   : patches randomly sampled per NCE layer (default 256)
        temperature   : PatchNCE temperature τ (default 0.07)
        lambda_nce    : weight on PatchNCE loss (default 1.0)
        lambda_idt    : weight on identity PatchNCE loss; 0 disables (default 1.0)
        lr            : Adam learning rate (default 2e-4)
        beta1         : Adam β₁ (default 0.5)
        n_epochs      : epochs with constant LR (default 200)
        n_epochs_decay: epochs over which LR linearly decays to 0 (default 200)
        use_amp       : enable mixed-precision training (default True)
    """

    def __init__(
        self,
        device: torch.device,
        in_channels: int = 1,
        ngf: int = 64,
        ndf: int = 64,
        n_blocks: int = 9,
        nce_layers: list[int] = [0, 4, 8, 12, 16],
        num_patches: int = 256,
        temperature: float = 0.07,
        lambda_nce: float = 1.0,
        lambda_idt: float = 1.0,
        lr: float = 2e-4,
        beta1: float = 0.5,
        n_epochs: int = 200,
        n_epochs_decay: int = 200,
        use_amp: bool = True,
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.n_blocks = n_blocks
        self.lambda_nce = lambda_nce
        self.lambda_idt = lambda_idt
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.use_amp = use_amp and device.type == "cuda"

        # ----- networks -----
        self.G : nn.Module = init_weights(ResnetGenerator(
            in_channels=in_channels, out_channels=in_channels,
            n_blocks=n_blocks,
        )).to(device)
        self.D_B = init_weights(
            PatchGANDiscriminator(in_channels=in_channels, ndf=ndf)
        ).to(device)

        nce_channels = self._nce_channel_sizes(ngf, in_channels, n_blocks)
        self.mlps = nn.ModuleList(
            [init_weights(PatchMLP(c)).to(device) for c in nce_channels]
        )

        # ----- losses -----
        self.crit_gan = LSGANLoss()
        self.crit_nce = [PatchNCELoss(temperature).to(device) for _ in nce_layers]

        # ----- optimizers (G, D, F separate — matches original CUT) -----
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_D = torch.optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_F = torch.optim.Adam(self.mlps.parameters(), lr=lr, betas=(beta1, 0.999))

        # Linear decay: constant for n_epochs, then decay to 0.
        # Original uses (n_epochs_decay + 1) in denominator and epoch_count=1.
        def _lr_lambda(epoch: int) -> float:
            return 1.0 - max(0, epoch + 1 - n_epochs) / (n_epochs_decay + 1)

        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, _lr_lambda)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, _lr_lambda)
        self.sched_F = torch.optim.lr_scheduler.LambdaLR(self.opt_F, _lr_lambda)

        # ----- AMP scalers -----
        self.scaler_G = GradScaler("cuda", enabled=self.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nce_channel_sizes(self, ngf: int, in_channels: int, n_blocks: int) -> list[int]:
        """Return output channel count at each nce_layer index.

        Flat generator layout with anti-aliased sampling (see ResnetGenerator):
          0           ReflectionPad2d                   → in_channels
          1-3         Conv(7×7)/Norm/ReLU               → ngf
          4-7         Conv(s=1)/Norm/ReLU/Downsample    → ngf×2
          8-11        Conv(s=1)/Norm/ReLU/Downsample    → ngf×4
          12..11+n    ResBlock ×n_blocks                → ngf×4
          ...         decoder layers
        """
        ch: dict[int, int] = {}
        ch[0] = in_channels
        for i in range(1, 4):
            ch[i] = ngf
        for i in range(4, 8):
            ch[i] = ngf * 2
        for i in range(8, 12):
            ch[i] = ngf * 4
        for i in range(12, 12 + n_blocks):
            ch[i] = ngf * 4
        return [ch[i] for i in self.nce_layers]

    @staticmethod
    def _sample_patches(
        feats: list[torch.Tensor], num_patches: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Sample spatial positions per image, keeping batch dimension.

        Returns:
            sampled_feats : list of [B*n, C] tensors  (n ≤ num_patches)
            patch_ids     : list of [n] index tensors (shared across batch), one per layer
        """
        out: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        for feat in feats:
            B, C, H, W = feat.shape
            # [B, C, H, W] → [B, H*W, C]
            flat = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
            S = flat.shape[1]  # H * W
            n = min(num_patches, S)
            idx = torch.randperm(S, device=feat.device)[:n]
            ids.append(idx)
            # Sample same positions from each image → [B, n, C] → [B*n, C]
            sampled = flat[:, idx, :].reshape(B * n, C)
            out.append(sampled)
        return out, ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def set_input(self, real_A: torch.Tensor, real_B: torch.Tensor):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        with autocast("cuda", enabled=self.use_amp):
            self.fake_B = self.G(self.real_A)
            if self.lambda_idt > 0.0:
                self.idt_B = self.G(self.real_B)

    # ------------------------------------------------------------------
    # NCE loss
    # ------------------------------------------------------------------

    def _compute_nce_loss(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean PatchNCE loss across nce_layers.

        src : source image (real)
        tgt : generated image (output of G applied to src)

        The encoder is run on both src and tgt at the same spatial patch
        positions. Running G's encoder on tgt (which is itself G(src))
        gives G's internal representation of the generated image — this is
        the correct query for the NCE objective, not a second full G pass.
        """
        B = src.shape[0]
        _, feats_src = self.G(src, self.nce_layers)
        _, feats_tgt = self.G(tgt, self.nce_layers)

        sampled_src, patch_ids = self._sample_patches(feats_src, self.num_patches)
        sampled_tgt: list[torch.Tensor] = []
        for feat, idx in zip(feats_tgt, patch_ids):
            Bf, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(Bf, -1, C)
            sampled_tgt.append(flat[:, idx, :].reshape(Bf * len(idx), C))

        total = torch.tensor(0.0, device=self.device)
        for feat_src, feat_tgt, mlp, crit in zip(
            sampled_src, sampled_tgt, self.mlps, self.crit_nce
        ):
            q = mlp(feat_tgt)   # query: from generated image
            k = mlp(feat_src)   # key  : from source image (same positions)
            total = total + crit(q, k, B)
        return total / len(self.nce_layers)

    # ------------------------------------------------------------------
    # Discriminator step
    # ------------------------------------------------------------------

    def _update_D(self) -> float:
        for p in self.D_B.parameters():
            p.requires_grad_(True)
        self.opt_D.zero_grad()

        with autocast("cuda", enabled=self.use_amp):
            loss_D_B = 0.5 * (
                self.crit_gan(self.D_B(self.real_B), True)
                + self.crit_gan(self.D_B(self.fake_B.detach()), False)
            )
        self.scaler_D.scale(loss_D_B).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        return loss_D_B.item()

    # ------------------------------------------------------------------
    # Generator step
    # ------------------------------------------------------------------

    def _update_G(self) -> dict:
        for p in self.D_B.parameters():
            p.requires_grad_(False)
        self.opt_G.zero_grad()
        self.opt_F.zero_grad()

        with autocast("cuda", enabled=self.use_amp):
            loss_adv = self.crit_gan(self.D_B(self.fake_B), True)

            # fake_B still has its computation graph; re-encoding through G's
            # encoder here is intentional and must NOT be detached.
            loss_nce = self._compute_nce_loss(self.real_A, self.fake_B) * self.lambda_nce

            loss_idt = torch.tensor(0.0, device=self.device)
            if self.lambda_idt > 0.0:
                loss_idt = self._compute_nce_loss(self.real_B, self.idt_B) * self.lambda_idt
                # Average NCE + identity NCE (matches original CUT)
                loss_nce_both = (loss_nce + loss_idt) * 0.5
            else:
                loss_nce_both = loss_nce

            loss_G = loss_adv + loss_nce_both

        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.step(self.opt_G)
        self.scaler_G.step(self.opt_F)
        self.scaler_G.update()

        return {
            "adv": loss_adv.item(),
            "nce": loss_nce.item(),
            "idt": loss_idt.item(),
            "G":   loss_G.item(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> dict:
        """Run one full optimization step. Returns scalar losses for logging."""
        self.forward()
        loss_D_B = self._update_D()
        losses_G = self._update_G()
        return {"D_B": loss_D_B, **losses_G}

    def scheduler_step(self):
        """Call once per epoch to advance LR schedulers."""
        self.sched_G.step()
        self.sched_D.step()
        self.sched_F.step()

    def save(self, path: str, epoch: int):
        torch.save(
            {
                "epoch":    epoch,
                "G":        self.G.state_dict(),
                "D_B":      self.D_B.state_dict(),
                "mlps":     self.mlps.state_dict(),
                "opt_G":    self.opt_G.state_dict(),
                "opt_D":    self.opt_D.state_dict(),
                "opt_F":    self.opt_F.state_dict(),
                "scaler_G": self.scaler_G.state_dict(),
                "scaler_D": self.scaler_D.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> int:
        """Load checkpoint; returns the saved epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D_B.load_state_dict(ckpt["D_B"])
        self.mlps.load_state_dict(ckpt["mlps"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        if "opt_F" in ckpt:
            self.opt_F.load_state_dict(ckpt["opt_F"])
        if "scaler_G" in ckpt:
            self.scaler_G.load_state_dict(ckpt["scaler_G"])
        if "scaler_D" in ckpt:
            self.scaler_D.load_state_dict(ckpt["scaler_D"])
        return ckpt["epoch"]
