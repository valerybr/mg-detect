"""Bilateral mammography dataset: paired left/right CC views, no-finding studies."""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BilateralDataset(Dataset):
    """Paired left/right CC mammogram dataset built from VinDr-Mammo annotations.

    Filters ``finding_annotations.csv`` to studies where **every** image is
    labelled ``'No Finding'``, then pairs the L CC and R CC images for each
    such study.

    Expected PNG layout::

        data_root/{study_id}/{image_id}.png

    Args:
        data_root:       Root directory containing converted PNG images.
        annotations_csv: Path to ``finding_annotations.csv``.
        split:           ``'training'``, ``'test'``, or ``None`` for all rows.
        img_size:        Side length to resize images to (default 512).
        flip_right:      Flip right-breast images horizontally so the nipple
                         faces left, matching the left-breast orientation.
                         Set to ``False`` if images were already flipped during
                         DICOM conversion (``--annotations`` flag in dicom_io).
    """

    def __init__(
        self,
        data_root: str,
        annotations_csv: str,
        split: str | None = None,
        img_size: int = 512,
        flip_right: bool = False,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.flip_right = flip_right

        self.pairs = self._build_pairs(Path(annotations_csv), split)

    # ------------------------------------------------------------------
    # Building the pair list
    # ------------------------------------------------------------------

    def _build_pairs(
        self, annotations_csv: Path, split: str | None
    ) -> list[tuple[Path, Path]]:
        # --- pass 1: collect all rows, optionally filtered by split ----------
        # study_id → {(laterality, view_position) → image_id}
        studies: dict[str, dict[tuple[str, str], str]] = {}
        # study_id → set of finding categories seen
        findings: dict[str, set[str]] = {}

        with open(annotations_csv, newline="") as f:
            for row in csv.DictReader(f):
                if split is not None and row["split"] != split:
                    continue
                sid = row["study_id"]
                findings.setdefault(sid, set()).add(row["finding_categories"])
                key = (row["laterality"], row["view_position"])
                studies.setdefault(sid, {})[key] = row["image_id"]

        # --- pass 2: keep only no-finding studies with both L CC and R CC ----
        pairs: list[tuple[Path, Path]] = []
        skipped: list[tuple[str, str]] = []  # (study_id, reason)

        for sid, images in studies.items():
            # Every row for this study must be "No Finding"
            if findings[sid] != {"['No Finding']"}:
                continue

            l_id = images.get(("L", "CC"))
            r_id = images.get(("R", "CC"))

            if l_id is None or r_id is None:
                missing = "L CC" if l_id is None else "R CC"
                skipped.append((sid, f"missing {missing}"))
                continue

            l_path = self.data_root / sid / f"{l_id}.png"
            r_path = self.data_root / sid / f"{r_id}.png"

            missing_files = [
                str(p) for p in (l_path, r_path) if not p.exists()
            ]
            if missing_files:
                skipped.append((sid, f"PNG not found: {', '.join(missing_files)}"))
                continue

            pairs.append((l_path, r_path))

        if skipped:
            print(f"[BilateralDataset] Skipped {len(skipped)} studies:")
            for sid, reason in skipped:
                print(f"  {sid}: {reason}")

        print(
            f"[BilateralDataset] Loaded {len(pairs)} paired studies"
            + (f" (split={split})" if split else "")
        )
        return pairs

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        l_path, r_path = self.pairs[idx]
        img_l = self._load(l_path, flip=False)
        img_r = self._load(r_path, flip=self.flip_right)
        return img_l, img_r

    def _load(self, path: Path, flip: bool) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        if flip:
            img = cv2.flip(img, 1)
        # [0, 255] uint8 → [-1, 1] float32, shape [1, H, W]
        tensor = torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0
        return tensor.unsqueeze(0)
