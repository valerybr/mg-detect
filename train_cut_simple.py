"""Training script for CUT (Contrastive Unpaired Translation).

Usage:
    python train_cut_simple.py --config configs/cut_defaults.yaml \\
        data.root=data/vindr-full/images-flip \\
        data.annotations=data/vindr/finding_annotations.csv \\
        output.dir=runs/cut_exp1

    # Override individual hyperparameters:
    python train_cut_simple.py --config configs/cut_defaults.yaml \\
        data.root=... data.annotations=... output.dir=... \\
        train.lr=0.001 model.ngf=32

All parameters are defined in configs/cut_defaults.yaml.
An experiment YAML (--config) overrides defaults; KEY=VALUE args override both.
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cut_model import CUTModel
from datasets import BilateralDataset

_DEFAULTS = Path(__file__).parent / "configs" / "cut_defaults.yaml"


def _load_cfg(config_path: str | None, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(_DEFAULTS)
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_readonly(cfg, True)
    return DictConfig(cfg)


def _validate(cfg: DictConfig):
    required = {
        "data.root":        cfg.data.root,
        "data.annotations": cfg.data.annotations,
        "output.dir":       cfg.output.dir,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Required config fields not set: {', '.join(missing)}")
    if cfg.data.split not in ("training", "test"):
        raise ValueError(f"data.split must be 'training' or 'test', got '{cfg.data.split}'")


def train(cfg: DictConfig):
    _validate(cfg)

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = BilateralDataset(
        data_root=cfg.data.root,
        annotations_csv=cfg.data.annotations,
        split=cfg.data.split,
        img_size=cfg.data.img_size,
        flip_right=cfg.data.flip_right,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} pairs  |  {len(loader)} batches/epoch")

    model = CUTModel(
        device=device,
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
    )

    start_epoch = 0
    if cfg.train.resume:
        start_epoch = model.load(cfg.train.resume)
        print(f"Resumed from {cfg.train.resume} (epoch {start_epoch})")

    total_epochs = cfg.train.n_epochs + cfg.train.n_epochs_decay
    for epoch in range(start_epoch, total_epochs):
        running = {k: 0.0 for k in ("D_B", "adv", "nce", "idt", "G")}

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
        for real_A, real_B in pbar:
            model.set_input(real_A, real_B)
            losses = model.optimize()
            for k in running:
                running[k] += losses[k]
            pbar.set_postfix(
                G=f"{losses['G']:.3f}",
                D=f"{losses['D_B']:.3f}",
                nce=f"{losses['nce']:.3f}",
            )

        n = len(loader)
        avg = {k: v / n for k, v in running.items()}
        print(
            f"[Epoch {epoch + 1:03d}] "
            f"G={avg['G']:.4f}  adv={avg['adv']:.4f}  "
            f"nce={avg['nce']:.4f}  idt={avg['idt']:.4f}  "
            f"D_B={avg['D_B']:.4f}"
        )

        model.scheduler_step()

        if (epoch + 1) % cfg.train.save_every == 0 or epoch + 1 == total_epochs:
            ckpt_path = output_dir / f"cut_ckpt_epoch_{epoch + 1:03d}.pt"
            model.save(str(ckpt_path), epoch + 1)
            print(f"  Saved {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CUT on bilateral mammograms")
    parser.add_argument("--config", default=None,
                        help="Path to experiment YAML (overrides cut_defaults.yaml)")
    parser.add_argument("overrides", nargs=argparse.REMAINDER,
                        help="KEY=VALUE overrides, e.g. train.lr=0.001 model.ngf=32")
    args = parser.parse_args()
    cfg = _load_cfg(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
