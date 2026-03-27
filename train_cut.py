"""Training script for CycleCUT.

Usage:
    python -m models.train_cut \\
        --data_root data/vindr-full/pngs \\
        --annotations data/vindr/finding_annotations.csv \\
        --output_dir runs/cut_exp1

Images are read as grayscale, normalised to [-1, 1], and resized to
``--img_size`` (default 512).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cut import CycleCUTModel
from datasets import BilateralDataset


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = BilateralDataset(
        data_root=args.data_root,
        annotations_csv=args.annotations,
        split=args.split,
        img_size=args.img_size,
        flip_right=args.flip_right,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} pairs  |  {len(loader)} batches/epoch")

    model = CycleCUTModel(
        device=device,
        nce_layers=args.nce_layers,
        num_patches=args.num_patches,
        lambda_nce=args.lambda_nce,
        lambda_cyc=args.lambda_cyc,
        lambda_idt=args.lambda_idt,
        lr=args.lr,
        n_epochs=args.n_epochs,
        n_epochs_decay=args.n_epochs_decay,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = model.load(args.resume)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    total_epochs = args.n_epochs + args.n_epochs_decay
    for epoch in range(start_epoch, total_epochs):
        running = {k: 0.0 for k in ("D_A", "D_B", "adv", "nce", "cyc", "idt", "G")}

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
        for real_A, real_B in pbar:
            model.set_input(real_A, real_B)
            losses = model.optimize()
            for k in running:
                running[k] += losses[k]
            pbar.set_postfix(
                G=f"{losses['G']:.3f}",
                D=f"{(losses['D_A'] + losses['D_B']):.3f}",
                nce=f"{losses['nce']:.3f}",
            )

        n = len(loader)
        avg = {k: v / n for k, v in running.items()}
        print(
            f"[Epoch {epoch + 1:03d}] "
            f"G={avg['G']:.4f}  adv={avg['adv']:.4f}  "
            f"nce={avg['nce']:.4f}  cyc={avg['cyc']:.4f}  "
            f"idt={avg['idt']:.4f}  "
            f"D_A={avg['D_A']:.4f}  D_B={avg['D_B']:.4f}"
        )

        model.scheduler_step()

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == total_epochs:
            ckpt_path = output_dir / f"ckpt_epoch_{epoch + 1:03d}.pt"
            model.save(str(ckpt_path), epoch + 1)
            print(f"  Saved {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Train CycleCUT on bilateral mammograms")
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing {study_id}/{image_id}.png files")
    parser.add_argument("--annotations", required=True,
                        help="Path to finding_annotations.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for checkpoints and logs")
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "test"],
                        help="Dataset split to use (default: training)")
    parser.add_argument("--flip_right", action="store_true",
                        help="Flip right-breast images horizontally at load time "
                             "(omit if already flipped during DICOM conversion)")
    parser.add_argument("--img_size",   type=int,   default=512)
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--n_epochs",       type=int,   default=100,
                        help="Epochs with constant LR")
    parser.add_argument("--n_epochs_decay", type=int,   default=100,
                        help="Epochs over which LR linearly decays to 0")
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lambda_nce", type=float, default=1.0)
    parser.add_argument("--lambda_cyc", type=float, default=10.0)
    parser.add_argument("--lambda_idt", type=float, default=5.0)
    parser.add_argument("--nce_layers", type=_parse_int_list, default=[3, 6, 9],
                        help="Encoder layer indices for PatchNCE, e.g. '0,4,8'")
    parser.add_argument("--num_patches",type=int,   default=256,
                        help="Patches sampled per NCE layer per image")
    parser.add_argument("--save_every", type=int,   default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
