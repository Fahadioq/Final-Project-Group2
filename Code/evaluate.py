"""Score a trained run on the evaluation sets of a split file. Owner: Modeling.

    # every test set in the split, clean images
    python Code/evaluate.py --run ~/of_data/runs/E0_resnet50_s0 --split ~/of_data/splits/E0/main.parquet

    # E4 robustness: also score every image after JPEG at these qualities
    python Code/evaluate.py --run ... --split ... --jpeg 95 85 75 65 50

    # E3 format flip: JPEG only the fake images, so a format shortcut would push them towards "real"
    python Code/evaluate.py --run ... --split ... --jpeg-fakes 75

    # only some sets
    python Code/evaluate.py --run ... --split ... --sets core_test reddit_test

Each scored set is added to the run's preds.parquet (common.schemas.PREDS_SCHEMA), one row per image, with these
names in the `set` column:
    <set>               clean images                       e.g. core_test
    <set>@jpeg<Q>       every image JPEG-compressed at Q    e.g. core_test@jpeg50
    <set>@fakejpeg<Q>   only fake images compressed at Q    e.g. core_test@fakejpeg75
Rows of a set that already exist are replaced, so re-running an evaluation is safe.

The model is used exactly as selected in train.py: weights from model.pt, the view and crop from model_card.json,
and the threshold tau fixed on the dev set. Nothing is tuned on a test set. Analysis & App computes the report's
metrics from preds.parquet (analysis/metrics.py); this script prints AUROC and accuracy at tau as a quick check
and stores them in metrics.json under "eval".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.gpu_aug import GpuAugment
from data.memmap_dataset import ViewDataset, make_loader, read_split
from models.runs import append_preds, auroc, load_run, predict, preds_table, write_json

TEST_PREFIX = "test:"


def eval_sets_in(split_path: Path | str) -> list[str]:
    """Names of the evaluation sets in a split file (roles "test:<name>"), sorted."""
    roles = read_split(split_path)["role"].unique()
    return sorted(r[len(TEST_PREFIX):] for r in roles if r.startswith(TEST_PREFIX))


def quick_check(scored: dict[str, np.ndarray], tau: float) -> dict:
    """AUROC and accuracy at the fixed tau: enough to spot a broken run, not the report's metrics."""
    labels, p = scored["label"].astype(bool), scored["p_fake"]
    return {"n": int(len(labels)), "n_fake": int(labels.sum()),
            "auroc": auroc(labels, p), "acc_at_tau": float(((p >= tau) == labels).mean())}


def evaluate(run_dir: Path, split_path: Path, sets: list[str] | None = None, jpeg: list[int] = (),
             jpeg_fakes: list[int] = (), view: str | None = None, batch_size: int = 256, workers: int = 2,
             data_root: Path | None = None, views_root: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, card = load_run(run_dir, device)
    aug = GpuAugment(device=device, crop=card["eval_crop"])
    available = eval_sets_in(split_path)
    sets = sets or available
    unknown = sorted(set(sets) - set(available))
    if unknown:
        raise ValueError(f"sets {unknown} are not in {split_path}; available: {available}")

    # (name in preds.parquet, JPEG quality or None, compress fakes only)
    variants = [("", None, False)] + [(f"@jpeg{q}", q, False) for q in jpeg] \
        + [(f"@fakejpeg{q}", q, True) for q in jpeg_fakes]
    results = {}
    for name in sets:
        ds = ViewDataset(split_path, [TEST_PREFIX + name], view=view or card["view"], classes=card["classes"],
                         data_root=data_root, views_root=views_root)
        loader = make_loader(ds, batch_size, train=False, num_workers=workers)
        for suffix, quality, fakes_only in variants:
            set_name = name + suffix
            scored = predict(model, loader, aug, device, jpeg_quality=quality, jpeg_fakes_only=fakes_only)
            append_preds(run_dir, preds_table(ds, scored, set_name, card["classes"]))
            results[set_name] = quick_check(scored, card["tau"])
            r = results[set_name]
            print(f"{set_name:<28} {r['n']:>7,} images ({r['n_fake']:,} fake) | AUROC {r['auroc']:.4f} | "
                  f"accuracy at tau {r['acc_at_tau']:.4f}")

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.setdefault("eval", {}).update(results)
    write_json(metrics_path, metrics)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory written by train.py")
    ap.add_argument("--split", required=True, help="split parquet holding the test:<set> roles")
    ap.add_argument("--sets", nargs="+", help="evaluation sets to score (default: every test set in the split)")
    ap.add_argument("--jpeg", nargs="+", type=int, default=[], metavar="Q", help="E4: also score at these qualities")
    ap.add_argument("--jpeg-fakes", nargs="+", type=int, default=[], metavar="Q",
                    help="E3 format flip: compress only the fakes at these qualities")
    ap.add_argument("--view", help="stored view to read (default: the view the run was trained on)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    for q in args.jpeg + args.jpeg_fakes:
        if not 1 <= q <= 100:
            ap.error(f"JPEG quality must be 1-100, got {q}")
    evaluate(Path(args.run).expanduser(), Path(args.split).expanduser(), args.sets, args.jpeg, args.jpeg_fakes,
             args.view, args.batch_size, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
