"""Prediction loop, dev-set metrics and run directories. Owner: Modeling.

Shared by train.py (model selection), evaluate.py (test sets) and inference.py (the app), so all three score
images the same way.

A run directory (common.schemas.RUN_FILES) holds:
  config.yaml      the settings the run was trained with
  metrics.json     per-epoch history, best epoch, dev AUROC, threshold, throughput
  preds.parquet    one row per scored image (common.schemas.PREDS_SCHEMA)
  model.pt         weights of the best epoch only
  model_card.json  what the app and evaluate.py need to rebuild and use the model

The metrics here are only the two that training needs (AUROC for early stopping, a threshold for the model card).
The full metric set for the report lives in analysis/metrics.py (Analysis & App role).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import schemas
from models.heads import build_detector, probabilities

UNKNOWN_GEN = "unknown-generator"
UNKNOWN_REAL = "unknown-real"


def auroc(labels, scores) -> float:
    """Area under the ROC curve: the probability that a random fake scores higher than a random real.

    Computed from ranks (Mann-Whitney U), ties counted as half. 0.5 = chance, 1.0 = perfect separation.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def youden_threshold(labels, scores) -> float:
    """Threshold tau that maximizes TPR - FPR (Youden's J) on the given set; predict fake when p_fake >= tau.

    Youden's J does not depend on the fake/real ratio, so the same rule works on balanced and unbalanced sets.
    tau is chosen once on the training distribution's dev set and then kept fixed on every test set:
    a moved threshold on unseen generators is part of what the experiments measure.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tpr = np.cumsum(y) / max(int(y.sum()), 1)
    fpr = np.cumsum(~y) / max(int((~y).sum()), 1)
    last_of_value = np.r_[s[1:] != s[:-1], True]           # one candidate per distinct score
    best = np.argmax((tpr - fpr)[last_of_value])
    return float(s[last_of_value][best])


@torch.no_grad()
def predict(model, loader, aug, device, jpeg_quality: int | None = None,
            jpeg_fakes_only: bool = False) -> dict[str, np.ndarray]:
    """Score every image in `loader` (evaluation mode: center crop, no randomness).

    jpeg_quality:    compress at this quality before scoring (E4 robustness sweep)
    jpeg_fakes_only: compress only the fake images (E3 format flip)
    Returns numpy arrays: idx (row in the dataset), label, p_fake, and attr_probs for the two-head model.
    """
    model.eval()
    use_amp = torch.device(device).type == "cuda"
    out: dict[str, list] = {"idx": [], "label": [], "p_fake": [], "attr_probs": []}
    for images, label, _attr, _mask, idx in loader:
        mask = label.bool() if jpeg_fakes_only else None
        x = aug.eval(images, jpeg_quality=jpeg_quality, jpeg_mask=mask)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            probs = probabilities(model(x))
        out["idx"].append(idx.numpy())
        out["label"].append(label.numpy())
        out["p_fake"].append(probs["p_fake"].cpu().numpy())
        if "attr_probs" in probs:
            out["attr_probs"].append(probs["attr_probs"].cpu().numpy())
    result = {k: np.concatenate(v) for k, v in out.items() if v}
    return result


def preds_table(dataset, scored: dict[str, np.ndarray], set_name: str, classes: list[str]) -> pa.Table:
    """Build a PREDS_SCHEMA table from predict() output, so Analysis & App can compute every metric."""
    n = len(scored["idx"])
    attr = scored.get("attr_probs")

    def column_of(name: str) -> np.ndarray:
        if attr is None or name not in classes:
            return np.full(n, np.nan, dtype=np.float32)
        return attr[:, classes.index(name)].astype(np.float32)

    table = pa.table({
        "row_id": dataset.rows["row_id"].to_numpy()[scored["idx"]],
        "set": [set_name] * n,
        "label": scored["label"].astype(np.int8),
        "p_fake": scored["p_fake"].astype(np.float32),
        "attr_probs": [list(r) for r in attr.astype(np.float32)] if attr is not None else [[] for _ in range(n)],
        "p_unknown_gen": column_of(UNKNOWN_GEN),
        "p_unknown_real": column_of(UNKNOWN_REAL),
    }, schema=schemas.PREDS_SCHEMA)
    schemas.validate(table, schemas.PREDS_SCHEMA)
    return table


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def save_checkpoint(path: Path, model, backbone: str, n_classes: int) -> None:
    """Weights plus the two numbers needed to rebuild the network."""
    torch.save({"state_dict": model.state_dict(), "backbone": backbone, "n_classes": n_classes}, path)


def load_run(run_dir: Path | str, device: torch.device | str = "cpu"):
    """Rebuild the trained detector from a run directory. Returns (model, model_card)."""
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "model_card.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = build_detector(ckpt["backbone"], ckpt["n_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), card


def append_preds(run_dir: Path, table: pa.Table) -> None:
    """Add rows to preds.parquet, replacing earlier rows of the same set (re-running an evaluation is safe)."""
    path = run_dir / "preds.parquet"
    if path.exists():
        old = pq.read_table(path)
        sets = set(table.column("set").to_pylist())
        keep = [s not in sets for s in old.column("set").to_pylist()]
        table = pa.concat_tables([old.filter(pa.array(keep)), table])
    pq.write_table(table, path)
