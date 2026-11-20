"""Filesystem locations shared by every role.

Override with environment variables:
  OF_DATA  durable data root on EBS        (default ~/of_data)
  OF_NVME  ephemeral instance-store mount   (default /mnt/nvme)
"""
from __future__ import annotations

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CODE_DIR.parent

OF_DATA = Path(os.environ.get("OF_DATA", Path.home() / "of_data"))
OF_NVME = Path(os.environ.get("OF_NVME", "/mnt/nvme"))

LANDING = OF_NVME / "landing"            # raw parquet shards, deleted after processing
MANIFEST_DIR = OF_DATA / "manifests"     # pass assignment, per-shard manifest, failures, STOP file
LOG_DIR = OF_DATA / "logs"
SPLITS_DIR = OF_DATA / "splits"          # split files (common.schemas.SPLIT_SCHEMA)
RUNS_DIR = OF_DATA / "runs"              # run directories (checkpoints are never committed)

RESULTS_DIR = REPO_DIR / "results"       # small CSV/JSON summaries, committed
CONFIG_DIR = CODE_DIR / "configs"
TAXONOMY_CSV = CODE_DIR / "taxonomy" / "generator_taxonomy.csv"


def index_dir(pool: str, root: Path | str | None = None) -> Path:
    return Path(root or OF_DATA) / "index" / pool


def index_path(pool: str, chunk: str, root: Path | str | None = None) -> Path:
    """One index parquet per processed shard ("chunk")."""
    return index_dir(pool, root) / f"{chunk}.parquet"


def view_path(view: str, pool: str, chunk: str, root: Path | str | None = None) -> Path:
    """uint8 array [n, 256, 256, 3] for one view of one chunk."""
    return Path(root or OF_DATA) / "views" / view / pool / f"{chunk}.npy"
