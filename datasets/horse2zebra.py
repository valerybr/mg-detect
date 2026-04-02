"""Unpaired horse↔zebra dataset for CUT / CycleGAN training."""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class Horse2ZebraDataset(Dataset):
    """Unpaired horse↔zebra image dataset.

    Images are loaded from ``{data_root}/{split}A/`` (horses) and
    ``{data_root}/{split}B/`` (zebras) independently.  Domains A and B are
    iterated in lock-step by index with cyclic wrapping so neither list runs
    out early when they differ in length.

    Training augmentation matches the original CUT paper:
    - Resize to ``img_size + 30`` (load size 286 for 256-px crops)
    - Random crop to ``img_size × img_size``
    - Random horizontal flip
    - Normalize to [-1, 1]

    Test transforms use a direct resize + center crop (no flip).

    Args:
        data_root: Root directory containing ``trainA``, ``trainB``,
                   ``testA``, ``testB`` sub-directories.
        split:     ``"train"`` or ``"test"``.
        img_size:  Output spatial resolution (default 256).
    """

    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_root: str, split: str = "train", img_size: int = 256):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
        self.split = split

        root = Path(data_root)
        self.files_A = sorted(
            p for p in (root / f"{split}A").iterdir() if p.suffix.lower() in self._EXTS
        )
        self.files_B = sorted(
            p for p in (root / f"{split}B").iterdir() if p.suffix.lower() in self._EXTS
        )

        if not self.files_A:
            raise FileNotFoundError(f"No images found in {root / f'{split}A'}")
        if not self.files_B:
            raise FileNotFoundError(f"No images found in {root / f'{split}B'}")

        load_size = img_size + 30  # 286 for img_size=256, matching CUT paper
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(load_size, InterpolationMode.BICUBIC),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                norm,
            ])

        print(
            f"[Horse2ZebraDataset] split={split} | "
            f"A={len(self.files_A)} images | B={len(self.files_B)} images"
        )

    def __len__(self) -> int:
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx: int):
        path_A = self.files_A[idx % len(self.files_A)]
        path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        img_A = self.transform(Image.open(path_A).convert("RGB"))
        img_B = self.transform(Image.open(path_B).convert("RGB"))
        return img_A, img_B
