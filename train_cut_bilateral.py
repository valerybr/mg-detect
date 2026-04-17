"""Training script for CUT (bilateral mammography).

Trains CUTModel (single generator G: A→B, PatchNCE loss, no cycle consistency)
on paired left/right CC mammograms from VinDr-Mammo using BilateralDataset.

Usage:
    python train_cut_bilateral.py --config configs/cut_bilateral.yaml \\
        data.root=/path/to/images data.annotations=/path/to/finding_annotations.csv \\
        output.dir=runs/cut_bilateral_exp1

    # Disable W&B:
    python train_cut_bilateral.py --config configs/cut_bilateral.yaml \\
        data.root=... data.annotations=... output.dir=... wandb.enabled=false

    # Resume from checkpoint:
    python train_cut_bilateral.py --config configs/cut_bilateral.yaml \\
        data.root=... data.annotations=... output.dir=runs/exp1 \\
        train.resume=runs/exp1/cut_ckpt_epoch_010.pt
"""

import argparse
import glob
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cut_model import CUTModel
from datasets import ScheduledBilateralDataset

_DEFAULTS = Path(__file__).parent / "configs" / "cut_bilateral.yaml"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg(config_path: str | None, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(_DEFAULTS)
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_readonly(cfg, True)
    return cfg


def _validate(cfg: DictConfig):
    missing = [
        k for k, v in {
            "data.root":        cfg.data.root,
            "data.annotations": cfg.data.annotations,
            "output.dir":       cfg.output.dir,
        }.items()
        if not v
    ]
    if missing:
        raise ValueError(f"Required config fields not set: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------

def _p_for_epoch(schedule, epoch: int) -> float:
    """Step schedule: list of (duration, p). Epochs past the end return 0.0."""
    cum = 0
    for dur, p in schedule:
        cum += int(dur)
        if epoch < cum:
            return float(p)
    return 0.0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_img(t: torch.Tensor) -> np.ndarray:
    """Tensor [1, H, W] in [-1, 1] → uint8 numpy [H, W]."""
    arr = t.squeeze().cpu().numpy()
    return ((arr + 1) * 127.5).clip(0, 255).astype(np.uint8)


def _save_samples(model, dataset, device, epoch: int, output_dir: Path, wandb_run):
    model.G.eval()
    with torch.no_grad():
        sA, sB = dataset[0]
        sA = sA.unsqueeze(0).to(device)
        sB = sB.unsqueeze(0).to(device)
        fake_B = model.G(sA)
        idt_B  = model.G(sB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(
        axes,
        [sA,      fake_B,         sB,      idt_B],
        ["realA (left)", "fakeB=G(A)", "realB (right)", "idt_B=G(B)"],
    ):
        ax.imshow(_to_img(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()

    out_path = output_dir / f"samples_epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    if wandb_run is not None:
        import wandb
        wandb_run.log({"samples": wandb.Image(fig)}, step=epoch)
    plt.close(fig)
    model.G.train()
    return out_path


def _sanity_check(model, dataset, device, epoch: int, avg_losses: dict,
                  output_dir: Path, wandb_run):
    warnings = []

    if avg_losses["D_B"] < 0.01:
        warnings.append(
            f"D_B={avg_losses['D_B']:.4f} (< 0.01) — discriminator may be collapsed"
        )

    model.G.eval()
    with torch.no_grad():
        sA, sB = dataset[0]
        sA = sA.unsqueeze(0).to(device)
        sB = sB.unsqueeze(0).to(device)
        fake_B = model.G(sA)
        idt_B  = model.G(sB)

    if fake_B.shape != sA.shape:
        warnings.append(f"Shape mismatch: input {sA.shape} → output {fake_B.shape}")

    fmin, fmax = fake_B.min().item(), fake_B.max().item()
    if fmin < -1.1 or fmax > 1.1:
        warnings.append(f"fakeB range [{fmin:.2f}, {fmax:.2f}] outside [-1, 1]")

    arr = fake_B[0].cpu().float()
    fft = torch.fft.fft2(arr)
    mag = torch.abs(fft)
    h, w = mag.shape[-2], mag.shape[-1]
    mask = torch.ones_like(mag, dtype=torch.bool)
    mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = False
    hf_ratio = mag[mask].sum() / mag.sum()
    if hf_ratio > 0.8:
        warnings.append(
            f"High-frequency energy ratio {hf_ratio:.2f} (> 0.8) — possible artifacts"
        )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(
        axes,
        [sA, fake_B, sB, idt_B],
        ["realA (left)", "fakeB=G(A)", "realB (right)", "idt_B=G(B)"],
    ):
        ax.imshow(_to_img(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    status = "OK" if not warnings else "WARNINGS"
    plt.suptitle(f"Sanity check — epoch {epoch} — {status}")
    plt.tight_layout()

    out_path = output_dir / f"sanity_epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    if wandb_run is not None:
        import wandb
        wandb_run.log({"sanity_check": wandb.Image(fig)}, step=epoch)
    plt.close(fig)
    model.G.train()

    if warnings:
        print(f"  SANITY CHECK (epoch {epoch}) — WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(
            f"  Sanity check passed (epoch {epoch}): "
            f"shape OK, range [{fmin:.2f}, {fmax:.2f}], "
            f"HF ratio {hf_ratio:.2f}, D_B={avg_losses['D_B']:.4f}"
        )


# ---------------------------------------------------------------------------
# W&B init
# ---------------------------------------------------------------------------

def _init_wandb(cfg: DictConfig, output_dir: Path):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping W&B logging. pip install wandb to enable.")
        return None

    run_name = cfg.wandb.run_name or output_dir.name
    existing = sorted(glob.glob(str(output_dir / "cut_ckpt_epoch_*.pt")))
    resume_mode = "must" if existing else "allow"

    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        resume=resume_mode,
        id=run_name,
        config={
            "dataset":           "vindr-bilateral",
            "img_size":          cfg.data.img_size,
            "batch_size":        cfg.train.batch_size,
            "n_epochs":          cfg.train.n_epochs,
            "n_epochs_decay":    cfg.train.n_epochs_decay,
            "lr":                cfg.train.lr,
            "beta1":             cfg.train.beta1,
            "lambda_nce":        cfg.train.lambda_nce,
            "lambda_idt":        cfg.train.lambda_idt,
            "nce_layers":        list(cfg.model.nce_layers),
            "num_patches":       cfg.model.num_patches,
            "temperature":       cfg.model.temperature,
            "use_amp":           cfg.train.use_amp,
            "save_images_every": cfg.train.save_images_every,
            "save_ckpt_every":   cfg.train.save_ckpt_every,
        },
    )
    print(f"W&B run: {run.url}  (resume={resume_mode})")
    return run


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: DictConfig):
    _validate(cfg)

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ----- dataset -----
    dataset = ScheduledBilateralDataset(
        data_root=cfg.data.root,
        annotations_csv=cfg.data.annotations,
        split=cfg.data.split,
        img_size=cfg.data.img_size,
        flip_right=cfg.data.flip_right,
        seed=cfg.train.random_seed,
    )
    schedule = [tuple(s) for s in cfg.train.random_schedule]
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Pairs        : {len(dataset.pairs)}")
    print(f"Dataset len  : {len(dataset)}")
    print(f"Batches/epoch: {len(loader)}")

    # ----- model -----
    model = CUTModel(
        device=device,
        in_channels=1,
        ngf=cfg.model.ngf,
        ndf=cfg.model.ndf,
        n_blocks=cfg.model.n_blocks,
        nce_layers=list(cfg.model.nce_layers),
        num_patches=cfg.model.num_patches,
        temperature=cfg.model.temperature,
        lambda_nce=cfg.train.lambda_nce,
        lambda_idt=cfg.train.lambda_idt,
        lr=cfg.train.lr,
        beta1=cfg.train.beta1,
        n_epochs=cfg.train.n_epochs,
        n_epochs_decay=cfg.train.n_epochs_decay,
        use_amp=cfg.train.use_amp,
    )

    def _count(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    print(f"G    params : {_count(model.G):.1f} M")
    print(f"D_B  params : {_count(model.D_B):.1f} M")
    print(f"MLPs params : {_count(model.mlps):.1f} M")
    print(f"Total       : {_count(model):.1f} M")

    # ----- resume -----
    start_epoch = 0
    if cfg.train.resume:
        start_epoch = model.load(cfg.train.resume)
        print(f"Resumed from {cfg.train.resume} — continuing from epoch {start_epoch + 1}")
    else:
        # Auto-detect latest checkpoint in output_dir
        existing = sorted(glob.glob(str(output_dir / "cut_ckpt_epoch_*.pt")))
        if existing:
            latest = existing[-1]
            start_epoch = model.load(latest)
            print(f"Auto-resumed from {latest} — continuing from epoch {start_epoch + 1}")
        else:
            print("No checkpoint found — training from scratch")

    # ----- W&B -----
    wandb_run = _init_wandb(cfg, output_dir)

    # ----- training loop -----
    total_epochs = cfg.train.n_epochs + cfg.train.n_epochs_decay
    epoch_times: list[float] = []

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        running = {k: 0.0 for k in ("D_B", "adv", "nce", "idt", "G")}

        p_random = _p_for_epoch(schedule, epoch)
        dataset.set_epoch_state(epoch, p_random)

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{total_epochs} [p_rand={p_random:.2f}]",
            leave=False,
        )
        for real_A, real_B in pbar:
            model.set_input(real_A, real_B)
            losses = model.optimize()
            for k in running:
                running[k] += losses[k]
            pbar.set_postfix(
                G=f"{losses['G']:.3f}",
                nce=f"{losses['nce']:.3f}",
                D=f"{losses['D_B']:.3f}",
            )

        model.scheduler_step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_epoch = sum(epoch_times[-5:]) / len(epoch_times[-5:])
        remaining = avg_epoch * (total_epochs - epoch - 1)

        n = len(loader)
        avg = {k: v / n for k, v in running.items()}

        print(
            f"[{epoch + 1:03d}/{total_epochs}] "
            f"G={avg['G']:.4f} adv={avg['adv']:.4f} "
            f"nce={avg['nce']:.4f} idt={avg['idt']:.4f} "
            f"D_B={avg['D_B']:.4f} "
            f"| {elapsed / 60:.1f} min/epoch "
            f"| ETA {remaining / 3600:.1f} h"
        )

        if wandb_run is not None:
            import wandb
            wandb_run.log(
                {f"loss/{k}": v for k, v in avg.items()} | {
                    "epoch":         epoch + 1,
                    "lr":            model.sched_G.get_last_lr()[0],
                    "p_random":      p_random,
                    "min_per_epoch": elapsed / 60,
                    "eta_hours":     remaining / 3600,
                },
                step=epoch + 1,
            )

        if epoch + 1 == 2:
            _sanity_check(model, dataset, device, epoch + 1, avg, output_dir, wandb_run)

        if (epoch + 1) % cfg.train.save_images_every == 0:
            out = _save_samples(model, dataset, device, epoch + 1, output_dir, wandb_run)
            print(f"  Saved samples → {out}")

        if (epoch + 1) % cfg.train.save_ckpt_every == 0 or epoch + 1 == total_epochs:
            ckpt_path = output_dir / f"cut_ckpt_epoch_{epoch + 1:03d}.pt"
            model.save(str(ckpt_path), epoch + 1)
            print(f"  Saved {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CUT on bilateral mammograms (local)"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to experiment YAML (overrides configs/cut_bilateral.yaml)",
    )
    parser.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="KEY=VALUE overrides, e.g. data.root=/data train.use_amp=true",
    )
    args = parser.parse_args()
    cfg = _load_cfg(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
