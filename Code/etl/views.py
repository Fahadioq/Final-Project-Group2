"""View recipes. Owner: Data. Imported by the ETL, the evaluation code, and the app.

Every image, real or fake, goes through exactly the same code path: same decoder, same
antialiased resampler (Lanczos with PIL's integer pre-reduction), same crop rule. Class-dependent
preprocessing would itself become a label cue.

  C  (controlled, main): if > 2 MP, downscale to 1 MP; then center-crop 256.
  N  (native):           center-crop 256 at original resolution (stored only when > 2 MP).
  R  (resized):          aspect-preserving resize to short side 256, center crop (approximates
                         OpenFake's SwinV2 preprocessing).
  Cxs (robustness):      downscale the full image by s, then the C recipe (evaluation pools only).
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 300_000_000  # Pexels photos reach ~45 MP

VIEW_SIZE = 256
MAX_MP = 2.0
TARGET_MP = 1.0
RESAMPLE = Image.LANCZOS
REDUCING_GAP = 3.0


def load_rgb(b: bytes) -> Image.Image:
    """Decode the first frame to 8-bit RGB. Alpha is dropped the same way for both classes."""
    im = Image.open(io.BytesIO(b))
    im.load()
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        return im.convert("RGBA").convert("RGB")
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        arr = np.clip(arr / (256.0 if arr.max() > 255 else 1.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "L").convert("RGB")
    return im.convert("RGB")


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, RESAMPLE, reducing_gap=REDUCING_GAP)


def capped_size(w: int, h: int, max_mp: float = MAX_MP, target_mp: float = TARGET_MP) -> tuple[int, int]:
    if w * h <= max_mp * 1e6:
        return w, h
    s = math.sqrt(target_mp * 1e6 / (w * h))
    return max(1, round(w * s)), max(1, round(h * s))


def needs_native(w: int, h: int, max_mp: float = MAX_MP) -> bool:
    """N differs from C only for images above the density cap."""
    return w * h > max_mp * 1e6


def eligibility(hf: dict, min_side: int = VIEW_SIZE, max_mp: float = MAX_MP,
                target_mp: float = TARGET_MP) -> tuple[bool, str]:
    """Can all views be built without upscaling? Returns (ok, skip_reason)."""
    if not hf.get("header_ok"):
        return False, "header_error"
    w, h = hf["width"], hf["height"]
    if min(w, h) < min_side:
        return False, "too_small"
    cw, ch = capped_size(w, h, max_mp, target_mp)
    if min(cw, ch) < min_side:
        return False, "too_small_after_cap"
    return True, ""


def density_cap(img: Image.Image, max_mp: float = MAX_MP, target_mp: float = TARGET_MP) -> Image.Image:
    size = capped_size(*img.size, max_mp, target_mp)
    return img if size == img.size else _resize(img, size)


def center_crop(img: Image.Image, size: int = VIEW_SIZE) -> np.ndarray:
    w, h = img.size
    if w < size or h < size:
        raise ValueError(f"image {w}x{h} smaller than crop {size}")
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)


def short_side_resize(img: Image.Image, short: int = VIEW_SIZE) -> Image.Image:
    w, h = img.size
    s = short / min(w, h)
    if s >= 1.0:  # never upscale; eligibility guarantees min side >= short
        return img
    return _resize(img, (max(short, round(w * s)), max(short, round(h * s))))


def view_c(img: Image.Image, size: int = VIEW_SIZE, max_mp: float = MAX_MP,
           target_mp: float = TARGET_MP) -> np.ndarray:
    return center_crop(density_cap(img, max_mp, target_mp), size)


def view_n(img: Image.Image, size: int = VIEW_SIZE) -> np.ndarray:
    return center_crop(img, size)


def view_r(img: Image.Image, size: int = VIEW_SIZE) -> np.ndarray:
    return center_crop(short_side_resize(img, size), size)


def view_c_scaled(img: Image.Image, factor: float, size: int = VIEW_SIZE, max_mp: float = MAX_MP,
                  target_mp: float = TARGET_MP) -> np.ndarray:
    """Robustness view: downscale the full image by `factor`, then apply the C recipe."""
    w, h = img.size
    small = _resize(img, (max(1, round(w * factor)), max(1, round(h * factor))))
    return view_c(small, size, max_mp, target_mp)
