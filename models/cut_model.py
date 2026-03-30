"""CUT model (Contrastive Unpaired Translation, Park et al. ECCV 2020).

One generator G: A→B, one discriminator D_B, PatchNCE loss + optional
identity PatchNCE loss. No cycle consistency, no second generator.
"""

import itertools

import torch
import torch.nn as nn

from .networks import ResnetGenerator, PatchGANDiscriminator, PatchMLP
from .cut import LSGANLoss, PatchNCELoss


class CUTModel(nn.Module):
    """CUT: Contrastive Unpaired Translation (Park et al., ECCV 2020).

    Trains a single generator G (A→B) with:
    - Adversarial loss (LSGAN): makes G(A) indistinguishable from real B
    - PatchNCE loss: patch-level mutual information between A and G(A)
    - Identity PatchNCE (optional): patch correspondence between B and G(B),
      encouraging content preservation when translating within domain B

    No cycle consistency loss, no second generator.

    Args:
        device        : torch device
        ngf           : base channel count for generator (default 64)
        ndf           : base channel count for discriminator (default 64)
        n_blocks      : residual blocks in generator bottleneck (default 9)
        nce_layers    : encoder layer indices for PatchNCE (default [3, 6, 9])
        num_patches   : patches randomly sampled per NCE layer (default 256)
        temperature   : PatchNCE temperature τ (default 0.07)
        lambda_nce    : weight on PatchNCE loss (default 1.0)
        lambda_idt    : weight on identity PatchNCE loss; 0 disables (default 1.0)
        lr            : Adam learning rate (default 2e-4)
        beta1         : Adam β₁ (default 0.5)
        n_epochs      : epochs with constant LR (default 200)
        n_epochs_decay: epochs over which LR linearly decays to 0 (default 200)
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
        lambda_idt: float = 1.0,
        lr: float = 2e-4,
        beta1: float = 0.5,
        n_epochs: int = 200,
        n_epochs_decay: int = 200,
    ):
        super().__init__()
        self.device = device
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.n_blocks = n_blocks
        self.lambda_nce = lambda_nce
        self.lambda_idt = lambda_idt
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay

        # ----- networks -----
        self.G = ResnetGenerator(n_blocks=n_blocks).to(device)
        self.D_B = PatchGANDiscriminator(ndf=ndf).to(device)

        nce_channels = self._nce_channel_sizes(ngf, n_blocks)
        self.mlps = nn.ModuleList([PatchMLP(c).to(device) for c in nce_channels])

        # ----- losses -----
        self.crit_gan = LSGANLoss()
        self.crit_nce = [PatchNCELoss(temperature).to(device) for _ in nce_layers]

        # ----- optimizers -----
        g_params = itertools.chain(self.G.parameters(), self.mlps.parameters())
        self.opt_G = torch.optim.Adam(g_params, lr=lr, betas=(beta1, 0.999))
        self.opt_D = torch.optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, 0.999))

        # Linear decay: constant for n_epochs, then decay to 0
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
        _, feats_src = self.G.forward_with_features(src, self.nce_layers)
        _, feats_tgt = self.G.forward_with_features(tgt, self.nce_layers)

        sampled_src, patch_ids = self._sample_patches(feats_src, self.num_patches)
        sampled_tgt: list[torch.Tensor] = []
        for feat, idx in zip(feats_tgt, patch_ids):
            B, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
            sampled_tgt.append(flat[idx])

        total = torch.tensor(0.0, device=self.device)
        for feat_src, feat_tgt, mlp, crit in zip(
            sampled_src, sampled_tgt, self.mlps, self.crit_nce
        ):
            q = mlp(feat_tgt)   # query: from generated image
            k = mlp(feat_src)   # key  : from source image (same positions)
            total = total + crit(q, k)
        return total / len(self.nce_layers)

    # ------------------------------------------------------------------
    # Discriminator step
    # ------------------------------------------------------------------

    def _update_D(self) -> float:
        for p in self.D_B.parameters():
            p.requires_grad_(True)
        self.opt_D.zero_grad()

        loss_D_B = 0.5 * (
            self.crit_gan(self.D_B(self.real_B), True)
            + self.crit_gan(self.D_B(self.fake_B.detach()), False)
        )
        loss_D_B.backward()
        self.opt_D.step()
        return loss_D_B.item()

    # ------------------------------------------------------------------
    # Generator step
    # ------------------------------------------------------------------

    def _update_G(self) -> dict:
        for p in self.D_B.parameters():
            p.requires_grad_(False)
        self.opt_G.zero_grad()

        loss_adv = self.crit_gan(self.D_B(self.fake_B), True)

        # fake_B still has its computation graph; re-encoding through G's
        # encoder here is intentional and must NOT be detached.
        loss_nce = self._compute_nce_loss(self.real_A, self.fake_B) * self.lambda_nce

        loss_idt = torch.tensor(0.0, device=self.device)
        if self.lambda_idt > 0.0:
            loss_idt = self._compute_nce_loss(self.real_B, self.idt_B) * self.lambda_idt

        loss_G = loss_adv + loss_nce + loss_idt
        loss_G.backward()
        self.opt_G.step()

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

    def save(self, path: str, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "G":     self.G.state_dict(),
                "D_B":   self.D_B.state_dict(),
                "mlps":  self.mlps.state_dict(),
                "opt_G": self.opt_G.state_dict(),
                "opt_D": self.opt_D.state_dict(),
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
        return ckpt["epoch"]
