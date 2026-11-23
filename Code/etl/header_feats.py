"""Header-only image features (no pixel decoding). Owner: Data.

Used for every streamed row (index), by the metadata-only shortcut baseline (E3.1), and by the
app's Shortcut lens. `PIL.Image.open` parses headers lazily, so this is cheap.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, JpegImagePlugin

Image.MAX_IMAGE_PIXELS = 300_000_000  # Pexels photos reach ~45 MP

_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)
_Q = np.arange(1, 101)
_SCALE = np.where(_Q < 50, 5000.0 / _Q, 200.0 - 2.0 * _Q)
# Sorted reference luminance tables for IJG quality 1..100: shape (100, 64)
_REF = np.sort(np.clip(np.floor((_STD_LUMA[None, :] * _SCALE[:, None] + 50) / 100), 1, 255), axis=1)
_SUBSAMPLING = {0: "444", 1: "422", 2: "420"}


def jpeg_quality(qtables) -> int | None:
    """IJG-equivalent quality whose luminance table best matches (order-insensitive)."""
    if not qtables:
        return None
    try:
        table = qtables[0]
        t = np.sort(np.asarray(list(table)[:64], dtype=float))
    except (KeyError, IndexError, TypeError):
        return None
    if t.size != 64:
        return None
    return int(_Q[np.abs(_REF - t).mean(axis=1).argmin()])


def empty_features(n_bytes: int = 0) -> dict:
    return dict(fmt=None, width=-1, height=-1, mode=None, file_bytes=n_bytes, jpeg_quality=None,
                jpeg_subsampling=None, progressive=False, has_exif=False, has_icc=False, n_frames=1,
                header_ok=False)


def header_features(b: bytes | None) -> dict:
    out = empty_features(len(b) if b else 0)
    if not b:
        return out
    try:
        with Image.open(io.BytesIO(b)) as im:
            out["fmt"] = im.format
            out["width"], out["height"] = im.size
            out["mode"] = im.mode
            info = im.info
            out["has_exif"] = bool(info.get("exif"))
            out["has_icc"] = bool(info.get("icc_profile"))
            out["n_frames"] = int(getattr(im, "n_frames", 1) or 1)
            if im.format == "JPEG":
                out["jpeg_quality"] = jpeg_quality(getattr(im, "quantization", None))
                try:
                    out["jpeg_subsampling"] = _SUBSAMPLING.get(JpegImagePlugin.get_sampling(im))
                except Exception:
                    pass
                out["progressive"] = bool(info.get("progressive") or info.get("progression"))
            out["header_ok"] = True
    except Exception:
        pass
    return out
