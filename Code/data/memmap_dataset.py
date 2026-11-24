"""Dataset over the chunked view store. Owner: Modeling.

The ETL (Data role) already decoded every kept image once and stored it as raw pixels:
    views/<view>/<pool>/<shard>.npy   uint8 [n, 256, 256, 3]
and wrote one index row per image saying where it lives (shard + `view_off`, or `n_off` for view N).

This module reads a split file (common.schemas.SPLIT_SCHEMA), joins it to the index to find each image's
(pool, shard, offset), and serves the pixels straight from the .npy files opened with mmap_mode='r'.
A memory map reads only the rows that are asked for, so the 100+ GB store never has to fit in RAM, and
no image is decoded during training: with 4 vCPUs on the instance, decoding would leave the GPU waiting.

One item = (image uint8 [256, 256, 3], label float 1=fake, attribution class index, bce_mask, row number).
Cropping, flipping, JPEG and normalization happen afterwards on the GPU (data/gpu_aug.py).

    python -m data.memmap_dataset                            # benchmark on a small synthetic store
    python -m data.memmap_dataset --split <split.parquet>    # benchmark on real views (run on the instance)
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from common import paths, schemas

INDEX_COLUMNS = ["row_id", "pool", "shard", "label", "model", "lineage", "view_off", "n_off"]


def read_split(split_path: Path | str, roles: list[str] | None = None) -> pd.DataFrame:
    """Read a split file and keep the requested roles ("train", "dev", "test:<set>", or "test" for every test set)."""
    table = pq.read_table(split_path)
    schemas.validate(table, schemas.SPLIT_SCHEMA)
    df = table.to_pandas()
    if roles:
        wanted = df["role"].isin(roles)
        if "test" in roles:
            wanted |= df["role"].str.startswith("test:")
        df = df[wanted]
    return df.reset_index(drop=True)


def attach_offsets(split: pd.DataFrame, data_root: Path | str | None = None) -> pd.DataFrame:
    """Add shard, view_off, n_off, label, model and lineage from the index to every split row.

    Only kept images (view_off >= 0) are read from the index. A split row with no kept image behind it is an
    error in the split file, so it raises instead of silently training on fewer images.
    """
    parts = []
    for pool in split["pool"].unique():
        index = pads.dataset(paths.index_dir(pool, data_root), format="parquet")
        table = index.to_table(columns=INDEX_COLUMNS, filter=pads.field("view_off") >= 0)
        parts.append(table.to_pandas())
    index_df = pd.concat(parts, ignore_index=True)
    merged = split.merge(index_df, on=["row_id", "pool"], how="left", validate="one_to_one")
    missing = merged["shard"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} split rows have no stored view, e.g. {merged.loc[missing, 'row_id'].iloc[0]}")
    merged["view_off"] = merged["view_off"].astype(np.int64)
    merged["n_off"] = merged["n_off"].astype(np.int64)
    return merged


def class_list(split: pd.DataFrame) -> list[str]:
    """Attribution classes, in a fixed order: the training roles' classes sorted by name.

    The order is saved in model_card.json["classes"], so predictions can be read back later.
    """
    train = split[split["role"].isin(["train", "dev"])]
    return sorted(train["attr_class"].dropna().unique().tolist())


class ViewDataset(Dataset):
    """Images and targets for one or more roles of a split file.

    split_path: split parquet (SPLIT_SCHEMA)
    roles:      e.g. ["train"], ["dev"], ["test:core_validation"], or ["test"] for every test set
    view:       "C" (main protocol), "R" (resized) or "N" (native; falls back to C for images <= 2 MP)
    classes:    attribution class order; pass the training run's list when evaluating
    data_root:  where the index lives (default OF_DATA, the durable EBS store)
    views_root: where the .npy views are read from (default OF_NVME after stage_data.sh, else OF_DATA)
    """

    def __init__(self, split_path: Path | str, roles: list[str], view: str = "C",
                 classes: list[str] | None = None, data_root: Path | str | None = None,
                 views_root: Path | str | None = None):
        if view not in schemas.VIEW_NAMES:
            raise ValueError(f"unknown view {view!r}, expected one of {schemas.VIEW_NAMES}")
        # The class order always comes from the training roles of the full split, whatever roles are loaded.
        self.classes = classes if classes is not None else class_list(read_split(split_path))
        self.rows = attach_offsets(read_split(split_path, roles), data_root)
        self.view = view
        self.views_root = Path(views_root) if views_root else default_views_root()

        # Plain numpy arrays, so a DataLoader worker reads them without touching pandas.
        class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.labels = (self.rows["label"] == "fake").to_numpy(np.float32)
        self.attr = self.rows["attr_class"].map(class_to_id).fillna(-1).to_numpy(np.int64)  # -1: not a known class
        self.bce_mask = self.rows["bce_mask"].to_numpy(bool)
        self.pools = self.rows["pool"].to_numpy()
        self.shards = self.rows["shard"].to_numpy()
        self.view_off = self.rows["view_off"].to_numpy(np.int64)
        self.n_off = self.rows["n_off"].to_numpy(np.int64)
        self._arrays: dict[tuple[str, str, str], np.ndarray] = {}   # opened lazily, once per worker process

    def __len__(self) -> int:
        return len(self.rows)

    def _array(self, view: str, pool: str, shard: str) -> np.ndarray:
        key = (view, pool, shard)
        if key not in self._arrays:
            self._arrays[key] = np.load(paths.view_path(view, pool, shard, self.views_root), mmap_mode="r")
        return self._arrays[key]

    def __getitem__(self, i: int):
        pool, shard = self.pools[i], self.shards[i]
        if self.view == "N" and self.n_off[i] >= 0:
            pixels = self._array("N", pool, shard)[self.n_off[i]]
        elif self.view == "N":                       # image <= 2 MP: N would equal C, so only C was stored
            pixels = self._array("C", pool, shard)[self.view_off[i]]
        else:
            pixels = self._array(self.view, pool, shard)[self.view_off[i]]
        image = torch.from_numpy(np.array(pixels))   # copy out of the memory map
        return image, self.labels[i], self.attr[i], self.bce_mask[i], i


def default_views_root() -> Path:
    """Read views from the NVMe copy when it exists (fast), otherwise from EBS."""
    return paths.OF_NVME if (paths.OF_NVME / "views").is_dir() else paths.OF_DATA


def make_loader(dataset: ViewDataset, batch_size: int = 128, train: bool = True,
                num_workers: int = 2, seed: int = 0) -> DataLoader:
    """DataLoader settings for the 4-vCPU instance: 2 workers, pinned memory for fast copies to the GPU."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,                              # equal-sized training batches
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def benchmark(loader: DataLoader, aug, n_batches: int) -> float:
    """Images per second for loading plus GPU augmentation, after one warm-up batch."""
    seen, start = 0, None
    for b, (images, *_rest) in enumerate(loader):
        x = aug.train(images)
        if x.is_cuda:
            torch.cuda.synchronize()
        if b == 0:
            start = time.perf_counter()               # skip worker start-up time
            continue
        seen += images.shape[0]
        if b >= n_batches:
            break
    return seen / (time.perf_counter() - start)


def main(argv: list[str] | None = None) -> int:
    from data.gpu_aug import GpuAugment
    from data.standin import make_standin_store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", help="split parquet; omit to benchmark a small synthetic store")
    ap.add_argument("--role", default="train")
    ap.add_argument("--view", default="C", choices=schemas.VIEW_NAMES)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        if args.split:
            split_path, root = args.split, None
        else:
            root = Path(tmp)
            split_path = make_standin_store(root, n_shards=4, rows_per_shard=400)
        ds = ViewDataset(split_path, [args.role], view=args.view, data_root=root, views_root=root)
        loader = make_loader(ds, args.batch_size, train=True, num_workers=args.workers)
        rate = benchmark(loader, GpuAugment(device=device), args.batches)
        print(f"{len(ds):,} images, view {args.view}, batch {args.batch_size}, {args.workers} workers, "
              f"device {device}: {rate:,.0f} images/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
