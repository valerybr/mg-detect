"""Train CUT (Contrastive Unpaired Translation) on the horse↔zebra dataset.

Usage:
    python train_cut_horse2zebra.py --data-root /path/to/horse2zebra --output-dir runs/h2z

    # Resume from checkpoint:
    python train_cut_horse2zebra.py --data-root ... --output-dir runs/h2z \\
        --resume runs/h2z/cut_ckpt_epoch_050.pt

Dataset layout expected:
    <data-root>/
        trainA/  (horses)
        trainB/  (zebras)
        testA/
        testB/
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.horse2zebra import Horse2ZebraDataset
from models.cut_model import CUTModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CUT on horse↔zebra")
    p.add_argument("--data-root", required=True, help="Path to horse2zebra dataset root")
    p.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-epochs", type=int, default=200, help="Epochs with constant LR")
    p.add_argument("--n-epochs-decay", type=int, default=200, help="Epochs for linear LR decay")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-nce", type=float, default=1.0)
    p.add_argument("--lambda-idt", type=float, default=1.0)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=9)
    p.add_argument("--num-patches", type=int, default=256)
    p.add_argument("--use-amp", action="store_true",
                   help="Enable bf16 AMP. Off by default — the original CUT recipe is fp32. "
                        "Only enable this for large datasets where memory is a constraint.")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="cut-horse2zebra")
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = Horse2ZebraDataset(args.data_root, split="train", img_size=args.img_size)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"Train dataset: {len(train_dataset)} samples  |  {len(loader)} batches/epoch")

    model = CUTModel(
        device=device,
        in_channels=3,  # RGB
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        num_patches=args.num_patches,
        lambda_nce=args.lambda_nce,
        lambda_idt=args.lambda_idt,
        lr=args.lr,
        n_epochs=args.n_epochs,
        n_epochs_decay=args.n_epochs_decay,
        use_amp=args.use_amp,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = model.load(args.resume)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    total_epochs = args.n_epochs + args.n_epochs_decay
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
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
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch + 1:03d}/{total_epochs}] "
            f"G={avg['G']:.4f}  adv={avg['adv']:.4f}  "
            f"nce={avg['nce']:.4f}  idt={avg['idt']:.4f}  "
            f"D_B={avg['D_B']:.4f}  "
            f"t={elapsed:.0f}s"
        )

        if run is not None:
            run.log({"epoch": epoch + 1, **{f"loss/{k}": v for k, v in avg.items()}})

        model.scheduler_step()

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == total_epochs:
            ckpt_path = output_dir / f"cut_ckpt_epoch_{epoch + 1:03d}.pt"
            model.save(str(ckpt_path), epoch + 1)
            print(f"  Saved {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
