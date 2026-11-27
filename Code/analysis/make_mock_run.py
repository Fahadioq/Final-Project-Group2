"""Write MOCK run directories and a matching MOCK index, so metrics and the app can be built before real runs exist.

Everything written here is synthetic. Run ids start with "MOCK-", model cards and metrics carry "mock": true,
and output goes to ~/of_data/mock/ (outside the repo). Never report these numbers as results.

    cd Code && python -m analysis.make_mock_run                       # -> ~/of_data/mock
    cd Code && python -m analysis.make_mock_run --out /tmp/mock --seeds 0 1 2

Layout (same as real data, under a different root):
    <out>/index/<pool>/mock-00000.parquet    common.schemas.INDEX_SCHEMA
    <out>/runs/<run_id>/                     config.yaml, metrics.json, preds.parquet, model_card.json
No model.pt is written: a mock run has no checkpoint (metrics.read_run reports has_checkpoint=False).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from common import paths, schemas

MOCK_ROOT = paths.OF_DATA / "mock"
CHUNK = "mock-00000"

# Stand-in taxonomy: raw model string -> lineage. Two names for one generator, as in the real data.
GENERATORS = {
    "sd-1.5": "sd12",
    "stable-diffusion-v1-5": "sd12",
    "sdxl": "sdxl",
    "juggernaut-xl": "sdxl",
    "flux.1-dev": "flux1",
    "flux.1-schnell": "flux1",
    "gpt-image-1": "proprietary",
}
REAL_SOURCES = ("laion", "pexels")

# preds set name -> index pool
SETS = {"dev": "core_train", "core_validation": "core_validation", "core_test": "core_test"}
FAKES_PER_GENERATOR = 120
REALS_PER_SOURCE = 300

# Mock runs: name -> training lineages (one in-distribution, one leave-one-lineage-out)
ALL_LINEAGES = sorted(set(GENERATORS.values()))
RUNS = {
    "e0-all": ALL_LINEAGES,
    "e2-holdout-flux1": [x for x in ALL_LINEAGES if x != "flux1"],
}


def make_index(seed: int = 0) -> dict[str, pa.Table]:
    """One INDEX_SCHEMA table per pool. Row ids are stable, so every run shares the same images."""
    rng = np.random.default_rng(seed)
    tables = {}
    for pool in dict.fromkeys(SETS.values()):
        rows = [("fake", m, lin) for m, lin in GENERATORS.items() for _ in range(FAKES_PER_GENERATOR)]
        rows += [("real", s, "real") for s in REAL_SOURCES for _ in range(REALS_PER_SOURCE)]
        n = len(rows)
        label = [r[0] for r in rows]
        fake = np.array([x == "fake" for x in label])
        row_id = [f"mock-{pool}-{i:06d}" for i in range(n)]
        width = np.where(fake, 1024, rng.integers(300, 4000, n)).astype(np.int32)
        height = np.where(fake, 1024, rng.integers(300, 3000, n)).astype(np.int32)
        cols = {
            "row_id": row_id,
            "image_sha256": [hashlib.sha256(r.encode()).hexdigest() for r in row_id],
            "pool": [pool] * n,
            "shard": [CHUNK] * n,
            "row_group": np.zeros(n, np.int32),
            "row_in_group": np.arange(n, dtype=np.int32),
            "label": label,
            "model": [r[1] for r in rows],
            "type": ["base"] * n,
            "release_date": [None] * n,
            "prompt_hash": [hashlib.sha256(f"prompt-{i % 50}".encode()).hexdigest()[:16] for i in range(n)],
            "lineage": [r[2] for r in rows],
            "bucket": [f"{r[0]}:{r[1]}" for r in rows],
            "fmt": np.where(fake, "PNG", "JPEG").tolist(),
            "width": width,
            "height": height,
            "mode": ["RGB"] * n,
            "file_bytes": (width.astype(np.int64) * height * np.where(fake, 1.2, 0.2)).astype(np.int64),
            "jpeg_quality": [None if f else 90 for f in fake],
            "jpeg_subsampling": [None if f else "420" for f in fake],
            "progressive": np.zeros(n, bool),
            "has_exif": (~fake) & (rng.random(n) < 0.3),
            "has_icc": np.zeros(n, bool),
            "n_frames": np.ones(n, np.int32),
            "header_ok": np.ones(n, bool),
            "eligible": np.ones(n, bool),
            "skip_reason": [None] * n,
            "hash_u": rng.random(n),
            "p_keep": np.ones(n, np.float32),
            "kept": np.ones(n, bool),
            "decode_ok": np.ones(n, bool),
            "view_off": np.arange(n, dtype=np.int32),
            "n_off": np.full(n, -1, np.int32),
        }
        table = pa.table(cols, schema=schemas.INDEX_SCHEMA)
        schemas.validate(table, schemas.INDEX_SCHEMA)
        tables[pool] = table
    return tables


def mock_scores(lineage: np.ndarray, fake: np.ndarray, train_lineages: list[str], rng) -> np.ndarray:
    """Synthetic p_fake: seen lineages separate well, held-out lineages less so."""
    seen = np.isin(lineage, train_lineages)
    p = np.where(~fake, rng.beta(2, 6, len(fake)),
                 np.where(seen, rng.beta(6, 2, len(fake)), rng.beta(3, 3.5, len(fake))))
    return np.round(p, 3).astype(np.float32)  # rounding creates tied scores


def make_run(index: dict[str, pa.Table], name: str, train_lineages: list[str], seed: int) -> dict:
    """Return the contents of one mock run directory (no checkpoint)."""
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    known = [m for m, lin in GENERATORS.items() if lin in train_lineages]
    classes = known + ["unknown-generator"] + list(REAL_SOURCES) + ["unknown-real"]

    parts = []
    for set_name, pool in SETS.items():
        t = index[pool].select(["row_id", "label", "model", "lineage"]).to_pandas()
        if set_name == "dev":  # dev only holds the run's own training lineages
            t = t[(t["label"] == "real") | t["lineage"].isin(train_lineages)]
        fake = (t["label"] == "fake").to_numpy()
        p_fake = mock_scores(t["lineage"].to_numpy(), fake, train_lineages, rng)

        # attribution: random, with extra weight on the true class
        attr = rng.dirichlet(np.ones(len(classes)), len(t))
        target = [classes.index(m) if m in classes else classes.index("unknown-generator") for m in t["model"]]
        attr[np.arange(len(t)), target] += 1.0
        attr = (attr / attr.sum(1, keepdims=True)).astype(np.float32)

        parts.append(pa.table({
            "row_id": t["row_id"].to_numpy(),
            "set": [set_name] * len(t),
            "label": fake.astype(np.int8),
            "p_fake": p_fake,
            "attr_probs": list(attr),
            "p_unknown_gen": attr[:, classes.index("unknown-generator")],
            "p_unknown_real": attr[:, classes.index("unknown-real")],
        }, schema=schemas.PREDS_SCHEMA))
    preds = pa.concat_tables(parts)
    schemas.validate(preds, schemas.PREDS_SCHEMA)

    # tau as train.py chooses it: Youden's J (max TPR - FPR) on the dev set
    dev = parts[0]
    y, p = dev.column("label").to_numpy(), dev.column("p_fake").to_numpy()
    fpr, tpr, thr = roc_curve(y, p)
    tau = float(thr[1:][np.argmax((tpr - fpr)[1:])])

    run_id = f"MOCK-{name}-s{seed}"
    card = {"run_id": run_id, "backbone": "resnet50", "classes": classes, "tau": tau,
            "view": "C", "eval_crop": 224, "mock": True}
    metrics = {"run_id": run_id, "best_epoch": None, "dev_auroc": float(roc_auc_score(y, p)), "tau": tau,
               "n_train": 0, "n_dev": dev.num_rows, "history": [], "gpu": None, "mock": True}
    config = {"mock": True, "backbone": "resnet50", "view": "C", "seed": seed, "train_lineages": train_lineages}
    return {"run_id": run_id, "preds": preds, "card": card, "metrics": metrics, "config": config}


def write_mock(out: Path | str = MOCK_ROOT, seeds=(0,)) -> list[Path]:
    """Write the mock index and every mock run under `out`. Returns the run directories."""
    out = Path(out)
    index = make_index(seed=0)
    for pool, table in index.items():
        path = paths.index_path(pool, CHUNK, root=out)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    run_dirs = []
    for seed in seeds:
        for name, lineages in RUNS.items():
            run = make_run(index, name, lineages, seed)
            run_dir = out / "runs" / run["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(run["preds"], run_dir / "preds.parquet")
            (run_dir / "model_card.json").write_text(json.dumps(run["card"], indent=2))
            (run_dir / "metrics.json").write_text(json.dumps(run["metrics"], indent=2))
            (run_dir / "config.yaml").write_text(yaml.safe_dump(run["config"], sort_keys=False))
            run_dirs.append(run_dir)
    return run_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(MOCK_ROOT), help="output root (default ~/of_data/mock)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="one run directory per seed and run")
    args = ap.parse_args()
    for run_dir in write_mock(args.out, args.seeds):
        print(f"MOCK run -> {run_dir}")
    print(f"MOCK index -> {Path(args.out) / 'index'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
