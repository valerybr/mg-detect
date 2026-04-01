import cv2
import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut, apply_windowing  # type: ignore
import argparse

def _resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Scale image to fit inside (size x size), pad right/bottom with zeros."""
    h, w = img.shape
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, :new_w] = img
    return canvas


def _crop_breast(img: np.ndarray) -> np.ndarray:
    """Remove background and crop to the breast bounding box."""
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return img[y : y + h, x : x + w]


def count_up_continuing_ones(b_arr):
    # indice continuing zeros from left side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    # from right side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def extract_breast(img):
    """ Used in Mammo-CLIP preprocessing """
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)  # To detect backgrounds easily
    height, _ = img.shape

    # whether each col is non-constant or not
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = count_up_continuing_ones(b_arr)
    # longest should be the breast
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    # whether each row is non-constant or not
    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = count_up_continuing_ones(b_arr)
    # longest should be the breast
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    return img_copy[row_ind][:, col_ind]



def load_mammogram(path: str, size: int = 512, flip: bool = False) -> np.ndarray:
    """Read a DICOM mammography file, remove background, crop breast, resize.

    Steps:
      1. Read pixel array and apply VOI LUT (window/level or LUT table)
      2. Invert MONOCHROME1 images so tissue is always bright
      3. Normalize to uint8
      4. Remove black background and crop to breast bounding box
      5. Optionally flip horizontally (e.g. to normalise right breasts to face left)
      6. Scale to fit (size x size) preserving aspect ratio, pad remainder with zeros

    Args:
        path: Path to a .dicom / .dcm file.
        size: Output side length in pixels (default 512).
        flip: If True, flip the image horizontally before resizing.

    Returns:
        uint8 numpy array of shape (size, size).
    """
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array  # keep integer dtype for apply_voi_lut

    # Apply VOI LUT: uses LUT table if present, falls back to WindowCenter/Width
    img = apply_voi_lut(pixels, ds).astype(np.float32)
    # img = apply_windowing(pixels, ds).astype(np.float32)

    # MONOCHROME1: 0 = white (air), max = black (tissue) — invert to MONOCHROME2 convention
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    # Normalize to [0, 255]
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    img = img.astype(np.uint8)

    # img = _crop_breast(img)
    img = extract_breast(img)
    if flip:
        img = cv2.flip(img, 1)
    img = _resize_with_pad(img, size)
    return img


if __name__ == "__main__":

    load_mammogram("data/1.dcm")

    import csv
    from pathlib import Path
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_path", help="path to dicom images", type=str, required=True)
    parser.add_argument("--output_path", help="path to output images", type=str, required=True)
    parser.add_argument("--size", help="output image size (default 512)", type=int, default=512)
    parser.add_argument(
        "--annotations",
        help="path to breast-level_annotations.csv; when provided, right-side (R) images are flipped horizontally",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    dicom_path = Path(args.dicom_path)
    output_path = Path(args.output_path)

    # Build image_id → laterality lookup from the annotations CSV.
    laterality: dict[str, str] = {}
    if args.annotations:
        with open(args.annotations, newline="") as f:
            for row in csv.DictReader(f):
                laterality[row["image_id"]] = row["laterality"]

    files = [p for p in dicom_path.rglob("*") if p.suffix.lower() in {".dcm", ".dicom"}]

    for src in tqdm(files, desc="Converting"):
        dst = output_path / src.relative_to(dicom_path).with_suffix(".png")
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        flip = laterality.get(src.stem) == "R"
        img = load_mammogram(str(src), size=args.size, flip=flip)
        cv2.imwrite(str(dst), img)
