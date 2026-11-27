"""Synthetic stand-in for the view store, so Modeling code runs before real split files exist. Owner: Modeling.

Writes, under one root directory, files with exactly the layout and schemas the ETL produces:
    index/<pool>/<shard>.parquet      common.schemas.INDEX_SCHEMA
    views/{C,R,N}/<pool>/<shard>.npy  uint8 [n, 256, 256, 3]
    splits/standin/standin.parquet    common.schemas.SPLIT_SCHEMA

The pixels are random noise, so nothing learned from them means anything about real images. Two details make
the store useful for checking code:
  - every stored image carries its own row number in the top-left pixel, so a test can confirm that the
    loader returned the image the index points to;
  - fake images get a faint 8x8 checkerboard added, so a training loop has a signal it can learn,
    and a falling loss shows that labels and images stay matched.

Only for testing the pipeline: no result computed on this data goes in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import paths, schemas

SIZE = schemas.VIEW_SIZE
FAKE_MODELS = ("sd-1.5", "sdxl", "flux.1-dev")
REAL_SOURCES = ("laion", "pexels")


def encode_id(pixels: np.ndarray, n: int) -> None:
    """Write the integer n into the top-left pixel (3 bytes, up to 16.7 million rows)."""
    pixels[0, 0] = (n >> 16 & 255, n >> 8 & 255, n & 255)


def decode_id(pixels) -> int:
    r, g, b = (int(v) for v in pixels[0, 0])
    return (r << 16) | (g << 8) | b


def _fake_pattern() -> np.ndarray:
    """A faint +-6 checkerboard of 8x8 squares: the learnable difference between synthetic reals and fakes."""
    yy, xx = np.indices((SIZE, SIZE))
    board = ((yy // 8 + xx // 8) % 2) * 12 - 6
    return board[..., None].astype(np.int16)


def _index_row(pool: str, shard: str, i: int, label: str, model: str, kept: bool,
               view_off: int, n_off: int, rng: np.random.Generator) -> dict:
    """One index row with every INDEX_SCHEMA column filled with a plausible value."""
    row_id = f"{pool}-{shard}-{i:05d}"
    return dict(
        row_id=row_id, image_sha256=f"{rng.integers(1 << 62):016x}", pool=pool, shard=shard,
        row_group=i // 70, row_in_group=i % 70, label=label, model=model, type="base", release_date="2024-01",
        prompt_hash=f"p{rng.integers(50)}", lineage="real" if label == "real" else model, bucket=f"{label}:{model}",
        fmt="JPEG" if label == "real" else "PNG", width=1024, height=1024, mode="RGB", file_bytes=500_000,
        jpeg_quality=90 if label == "real" else None, jpeg_subsampling="420" if label == "real" else None,
        progressive=False, has_exif=False, has_icc=False, n_frames=1, header_ok=True,
        eligible=kept, skip_reason="" if kept else "not_selected", hash_u=float(rng.random()),
        p_keep=0.8, kept=kept, decode_ok=kept, view_off=view_off, n_off=n_off,
    )


def _role(u: float, test_sets: tuple[str, ...]) -> str:
    """10% dev; with test sets, the next 20% is spread over them; the rest train. One random draw per row."""
    if u < 0.1:
        return "dev"
    if test_sets and u < 0.3:
        return f"test:{test_sets[int((u - 0.1) / 0.2 * len(test_sets))]}"
    return "train"


def make_standin_store(root: Path | str, n_shards: int = 3, rows_per_shard: int = 200,
                       pool: str = "core_train", seed: int = 0, test_sets: tuple[str, ...] = ()) -> Path:
    """Build the synthetic store under `root` and return the path of its split file.

    test_sets: names of evaluation sets to carve out of the store (role "test:<name>"), for checking
    evaluate.py. Empty by default, so the store is identical to earlier versions.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    pattern = _fake_pattern()
    split_rows, next_id = [], 0

    for s in range(n_shards):
        shard = f"train-{s:05d}-of-standin"
        labels = rng.choice(["real", "fake"], size=rows_per_shard)
        kept = rng.random(rows_per_shard) < 0.8                    # like the ETL, not every row is kept
        big = kept & (rng.random(rows_per_shard) < 0.3)           # about 30% "over 2 MP": they also get an N view
        n_kept, n_big = int(kept.sum()), int(big.sum())
        views = {v: np.zeros((n, SIZE, SIZE, 3), np.uint8) for v, n in (("C", n_kept), ("R", n_kept), ("N", n_big))}

        index_rows, vi, ni = [], 0, 0
        for i in range(rows_per_shard):
            label = str(labels[i])
            model = str(rng.choice(FAKE_MODELS if label == "fake" else REAL_SOURCES))
            view_off = n_off = -1
            if kept[i]:
                view_off = vi
                for v in ("C", "R"):
                    img = rng.integers(0, 256, (SIZE, SIZE, 3)).astype(np.int16)
                    if label == "fake":
                        img += pattern
                    views[v][vi] = np.clip(img, 0, 255).astype(np.uint8)
                    encode_id(views[v][vi], next_id)
                if big[i]:
                    n_off = ni
                    views["N"][ni] = views["C"][vi]
                    encode_id(views["N"][ni], next_id)
                    ni += 1
                vi += 1
                split_rows.append(dict(
                    row_id=f"{pool}-{shard}-{i:05d}", pool=pool,
                    role=_role(rng.random(), test_sets),
                    attr_class=model, bce_mask=True,
                ))
                next_id += 1
            index_rows.append(_index_row(pool, shard, i, label, model, bool(kept[i]), view_off, n_off, rng))

        index_path = paths.index_path(pool, shard, root)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(index_rows, schema=schemas.INDEX_SCHEMA)
        schemas.validate(table, schemas.INDEX_SCHEMA)
        pq.write_table(table, index_path)
        for v, arr in views.items():
            if len(arr):
                path = paths.view_path(v, pool, shard, root)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr)

    split_path = root / "splits" / "standin" / "standin.parquet"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split = pa.Table.from_pylist(split_rows, schema=schemas.SPLIT_SCHEMA)
    schemas.validate(split, schemas.SPLIT_SCHEMA)
    pq.write_table(split, split_path)
    return split_path
