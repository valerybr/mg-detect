"""Loss functions for CycleCUT model."""

import torch
import torch.nn as nn


class LSGANLoss(nn.Module):
    """Least-squares GAN loss (Mao et al., 2017).

    For the generator  : target = 1  (wants discriminator to output 1 for fakes)
    For the discriminator: real target = 1, fake target = 0
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def __call__(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)


class PatchNCELoss(nn.Module):
    """Patch-level noise-contrastive estimation loss (Park et al., ECCV 2020).

    For each query patch q at position (i,j):
      positive  = key at the same (i,j) in the source image
      negatives = all other patches *from the same image*

    Temperature τ = 0.07 (from the CUT paper).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, feat_q: torch.Tensor, feat_k: torch.Tensor, batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            feat_q     : [B*n, C] — query embeddings (from generated image), L2-normed
            feat_k     : [B*n, C] — key embeddings (from source image, same positions), L2-normed
            batch_size : B — number of images; negatives are computed per-image
        Returns:
            scalar NCE loss (mean over all samples)
        """
        feat_k = feat_k.detach()

        # Force float32 for numerical stability under AMP.
        # The bmm + division by temperature (0.07 → ×14.3) can overflow float16.
        feat_q = feat_q.float()
        feat_k = feat_k.float()

        dim = feat_q.shape[1]
        N = feat_q.shape[0]          # B * n
        n = N // batch_size           # patches per image

        # Positive logits: element-wise dot → [B*n, 1]
        l_pos = (feat_q * feat_k).sum(dim=1, keepdim=True)

        # Negative logits: per-image via bmm → [B, n, n]
        l_neg = torch.bmm(
            feat_q.view(batch_size, n, dim),
            feat_k.view(batch_size, n, dim).transpose(2, 1),
        )
        # Mask diagonal so k_i is not used as a negative for query q_i
        diag = torch.eye(n, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg.masked_fill_(diag, -10.0)
        l_neg = l_neg.view(N, n)      # [B*n, n]

        # [B*n, 1+n]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(N, dtype=torch.long, device=feat_q.device)
        return self.cross_entropy(logits, labels).mean()
