"""Report metrics for run directories: balanced per-generator sets, F1 at the run's fixed tau, AUROC,
TPR at 5% FPR, balanced accuracy, oracle-tau F1, bootstrap CIs, generalization gaps, mean/sd over seeds.

    cd Code && python -m analysis.metrics --run ~/of_data/mock/runs/MOCK-e0-all-s0 --index-root ~/of_data/mock

Predictions are joined to the index on row_id to get each image's generator and lineage. tau comes from
model_card.json; an image is predicted fake when p_fake >= tau (same rule as models/runs.py).
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from common import paths, schemas

METRICS = ("f1", "auroc", "tpr_at_5fpr", "balanced_acc", "oracle_f1")
FPR_TARGET = 0.05
N_BOOT = 1000
MIN_PER_CLASS = 30  # groups below this are flagged "small", not dropped


# ---- checks -----------------------------------------------------------------------------------------------

def check_tau(tau) -> float:
    """tau must be a finite number in [0, 1]."""
    tau = float(tau)
    if not (np.isfinite(tau) and 0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be a finite number in [0, 1], got {tau}")
    return tau


def check_scores(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Labels must be 0/1 and scores finite probabilities in [0, 1]."""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"labels and scores differ in shape: {y.shape} vs {p.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (real) or 1 (fake)")
    if not np.isfinite(p).all():
        raise ValueError("p_fake contains NaN or infinite values")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p_fake must be in [0, 1]")
    return y.astype(int), p


def has_both_classes(y) -> bool:
    """True when y holds at least one fake (1) and one real (0)."""
    y = np.asarray(y)
    return bool((y == 1).any() and (y == 0).any())


# ---- loading ----------------------------------------------------------------------------------------------

def read_run(run_dir: Path | str) -> dict:
    """Read a run directory. model.pt is optional (mock runs have none); every other run file is required."""
    run_dir = Path(run_dir)
    missing = [f for f in schemas.RUN_FILES if f != "model.pt" and not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing {missing}")
    table = pq.read_table(run_dir / "preds.parquet")
    schemas.validate(table, schemas.PREDS_SCHEMA)
    card = json.loads((run_dir / "model_card.json").read_text())
    absent = [k for k in schemas.MODEL_CARD_KEYS if k not in card]
    if absent:
        raise ValueError(f"{run_dir}: model_card.json missing {absent}")
    card["tau"] = check_tau(card["tau"])
    preds = table.to_pandas()
    check_scores(preds["label"], preds["p_fake"])
    return {
        "run_id": card["run_id"],
        "preds": preds,
        "card": card,
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
        "config": yaml.safe_load((run_dir / "config.yaml").read_text()),
        "has_checkpoint": (run_dir / "model.pt").exists(),
        "mock": bool(card.get("mock", False)),
    }


def read_index(root: Path | str | None = None, pools=None,
               columns=("row_id", "pool", "label", "model", "lineage")) -> pd.DataFrame:
    """Read index files under <root>/index/ (all pools, or the given ones). row_id must be unique."""
    base = Path(root or paths.OF_DATA) / "index"
    pools = pools or sorted(p.name for p in base.iterdir() if p.is_dir())
    files = [f for pool in pools for f in sorted(paths.index_dir(pool, root).glob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"no index files under {base} for pools {pools}")
    index = pd.concat([pq.read_table(f, columns=list(columns)).to_pandas() for f in files], ignore_index=True)
    dup = index["row_id"].duplicated()
    if dup.any():
        raise ValueError(f"index has {int(dup.sum())} duplicate row_ids, e.g. {index.loc[dup, 'row_id'].iloc[0]}")
    return index


def attach_index(preds: pd.DataFrame, index: pd.DataFrame, group_map: dict | None = None) -> pd.DataFrame:
    """Join predictions to the index on row_id and add a `generator` column.

    Raises on duplicate (set, row_id) predictions, row_ids missing from the index, or mismatched labels.
    group_map (raw model string -> canonical name) merges duplicate names; unmapped strings stay as they are.
    """
    dup = preds.duplicated(["set", "row_id"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate predictions (same set and row_id)")
    missing = ~preds["row_id"].isin(index["row_id"])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} predicted row_ids not in the index, e.g. "
                         f"{preds.loc[missing, 'row_id'].iloc[0]}")
    df = preds.merge(index.rename(columns={"label": "index_label"}), on="row_id", how="left",
                     validate="many_to_one")
    bad = df["label"].astype(int) != (df["index_label"] == "fake").astype(int)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows where the predicted label disagrees with the index label")
    model = df["model"]
    df["generator"] = model.map(group_map).fillna(model) if group_map else model
    return df.drop(columns="index_label")


# ---- metrics on one set (labels: 1 = fake) ----------------------------------------------------------------

def tpr_at_fpr(y, p, fpr_target: float = FPR_TARGET) -> float:
    """Highest TPR among thresholds with FPR <= fpr_target."""
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return float(tpr[fpr <= fpr_target].max())


def oracle_f1(y, p) -> float:
    """Best F1 over all thresholds on this set."""
    precision, recall, _ = precision_recall_curve(y, p)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return float(f1.max())


def score(y, p, tau: float) -> dict:
    """All point metrics; NaN when a class is missing."""
    y, p = check_scores(y, p)
    tau = check_tau(tau)
    if not has_both_classes(y):
        return {m: float("nan") for m in METRICS}
    pred = (p >= tau).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0.0)),
        "auroc": float(roc_auc_score(y, p)),
        "tpr_at_5fpr": tpr_at_fpr(y, p),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "oracle_f1": oracle_f1(y, p),
    }


def stratified_resample(y, rng) -> np.ndarray:
    """Bootstrap indices drawn with replacement within each class (class counts unchanged)."""
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, p, tau: float, n_boot: int = N_BOOT, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile CI for every metric from stratified resamples; deterministic for a given seed."""
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    y, p = check_scores(y, p)
    tau = check_tau(tau)
    if not has_both_classes(y):
        return {m: (float("nan"), float("nan")) for m in METRICS}
    rng = np.random.default_rng(seed)
    draws = pd.DataFrame([score(y[i], p[i], tau) for i in (stratified_resample(y, rng) for _ in range(int(n_boot)))])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {m: (float(np.nanpercentile(draws[m], lo)), float(np.nanpercentile(draws[m], hi))) for m in METRICS}


# ---- balanced per-group evaluation ------------------------------------------------------------------------

def balanced_set(df: pd.DataFrame, group: str, by: str = "generator", seed: int = 0) -> tuple[pd.DataFrame, int]:
    """One group's fakes plus an equal number of reals from the same rows.

    Fakes are subsampled when reals are fewer. Rows are sorted by row_id first, so the sample depends only on
    the seed, the group name and the row ids, not on input order. Returns (rows, fakes_dropped).
    """
    if "row_id" in df:
        df = df.sort_values("row_id", kind="stable")
    rng = np.random.default_rng([seed, zlib.crc32(str(group).encode())])
    fakes = df[(df["label"] == 1) & (df[by] == group)]
    reals = df[df["label"] == 0]
    n = min(len(fakes), len(reals))
    take_f = fakes.iloc[np.sort(rng.choice(len(fakes), n, replace=False))] if n < len(fakes) else fakes
    take_r = reals.iloc[np.sort(rng.choice(len(reals), n, replace=False))]
    return pd.concat([take_f, take_r]), len(fakes) - n


def evaluate_groups(df: pd.DataFrame, tau: float, by: str = "generator", sets=None,
                    n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """One row per (set, fake group): counts, point metrics and bootstrap CIs on a balanced set."""
    tau = check_tau(tau)
    rows = []
    for set_name in sets or sorted(df["set"].unique()):
        part = df[df["set"] == set_name]
        for group in sorted(part.loc[part["label"] == 1, by].unique()):
            bal, dropped = balanced_set(part, group, by=by, seed=seed)
            y, p = bal["label"].to_numpy(), bal["p_fake"].to_numpy()
            n_fake, n_real = int(y.sum()), int(len(y) - y.sum())
            row = {"set": set_name, "group_by": by, "group": group, "n_fake": n_fake, "n_real": n_real,
                   "fakes_dropped": dropped, "small": min(n_fake, n_real) < MIN_PER_CLASS}
            row.update(score(y, p, tau))
            for m, (lo, hi) in bootstrap_ci(y, p, tau, n_boot=n_boot, seed=seed).items():
                row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_reference(df: pd.DataFrame, tau: float, set_name: str, groups=None, by: str = "lineage",
                     seed: int = 0) -> dict:
    """Point metrics on one set, pooling the fakes of `groups` (default all) against an equal number of reals."""
    part = df[df["set"] == set_name]
    if groups is not None:
        part = part[(part["label"] == 0) | part[by].isin(groups)]
    part = part.assign(_all=np.where(part["label"] == 1, "fakes", "reals"))
    bal, _ = balanced_set(part, "fakes", by="_all", seed=seed)
    return score(bal["label"], bal["p_fake"], tau)


def generalization_gap(table: pd.DataFrame, reference, metrics=("f1", "auroc"), on: str = "group") -> pd.DataFrame:
    """Add gap_<metric> = reference - value (positive = worse than the reference).

    reference: a dict of scalars (e.g. from pooled_reference) or an evaluate_groups table matched on `on`.
    """
    out = table.copy()
    if isinstance(reference, pd.DataFrame):
        ref = reference.set_index(on)
        if ref.index.duplicated().any():
            raise ValueError("reference table has repeated groups; filter it to one set first")
        for m in metrics:
            out[f"gap_{m}"] = out[on].map(ref[m]) - out[m]
    else:
        for m in metrics:
            out[f"gap_{m}"] = reference[m] - out[m]
    return out


def mean_sd(tables: list[pd.DataFrame], seeds: list[int], keys=("set", "group_by", "group"),
            metrics=METRICS) -> pd.DataFrame:
    """Mean and sample sd of each metric across seeds; one evaluate_groups table per seed."""
    if len(tables) != len(seeds):
        raise ValueError(f"{len(tables)} tables but {len(seeds)} seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    allt = pd.concat([t.assign(seed=s) for t, s in zip(tables, seeds)], ignore_index=True)
    if allt.duplicated([*keys, "seed"]).any():
        raise ValueError("a table has repeated rows for the same keys")
    g = allt.groupby(list(keys))[list(metrics)]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    out["n_seeds"] = g.size()
    return out.reset_index()


# ---- command line -----------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, nargs="+", help="run directories")
    ap.add_argument("--index-root", default=None, help="root holding index/<pool>/ (default ~/of_data)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = read_index(args.index_root)
    for run_dir in map(Path, args.run):
        run = read_run(run_dir)
        df = attach_index(run["preds"], index)
        out = run_dir / "analysis"
        out.mkdir(exist_ok=True)
        for by in ("generator", "lineage"):
            table = evaluate_groups(df, run["card"]["tau"], by=by, n_boot=args.n_boot, seed=args.seed)
            table.insert(0, "run_id", run["run_id"])
            table.insert(1, "mock", run["mock"])
            table.to_csv(out / f"per_{by}.csv", index=False)
        tag = "MOCK run (synthetic numbers)" if run["mock"] else "run"
        ckpt = "" if run["has_checkpoint"] else ", no checkpoint"
        print(f"{tag} {run['run_id']}{ckpt} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
