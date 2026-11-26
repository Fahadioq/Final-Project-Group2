"""Checks for train.py and models/runs.py (no pytest, no GPU, no weight download).

    cd Code && python -m tests.test_train

The end-to-end check trains a small ResNet-18 for two steps on the synthetic store, so it finishes on a laptop CPU.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from common import schemas
from data.gpu_aug import GpuAugment
from data.memmap_dataset import ViewDataset, make_loader
from data.standin import make_standin_store
from models.runs import auroc, load_run, predict, youden_threshold
from train import DEFAULT_CONFIG, load_config, lr_schedule, train


def test_auroc_known_values():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5          # all tied: chance
    assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75       # one of four pairs out of order
    assert np.isnan(auroc([1, 1], [0.2, 0.3]))                      # one class only: undefined


def test_youden_threshold_separates_clean_data():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    tau = youden_threshold(labels, scores)
    assert ((scores >= tau) == labels.astype(bool)).all() and tau == 0.6


def test_lr_schedule_warms_up_then_decays():
    f = lr_schedule(steps_per_epoch=10, total_steps=50, warmup_epochs=1)
    values = [f(s) for s in range(50)]
    assert values[0] == 0.1 and values[9] == 1.0                    # linear warm-up over the first epoch
    assert all(a >= b for a, b in zip(values[9:], values[10:]))     # then never increases
    assert values[-1] < 0.01


def test_config_overrides():
    cfg = load_config(DEFAULT_CONFIG, ["epochs=3", "lr_backbone=2e-4", "two_heads=false", "backbone=convnext_tiny"])
    assert cfg["epochs"] == 3 and cfg["lr_backbone"] == 2e-4 and cfg["two_heads"] is False
    try:
        load_config(DEFAULT_CONFIG, ["epocs=3"])
    except KeyError:
        pass
    else:
        raise AssertionError("a misspelled setting must be rejected")


def _tiny_run(tmp: Path, two_heads: bool) -> tuple[Path, Path, dict]:
    store = tmp / "store"
    split = make_standin_store(store, n_shards=2, rows_per_shard=40, seed=0)
    cfg = load_config(DEFAULT_CONFIG, ["backbone=resnet18", "pretrained=false", "epochs=2", "batch_size=8",
                                       "eval_batch_size=16", "workers=0", f"two_heads={str(two_heads).lower()}"])
    run_dir = tmp / "runs" / ("two" if two_heads else "binary")
    metrics = train(cfg, split, run_dir, data_root=store, views_root=store, max_steps=2)
    return split, run_dir, metrics


def test_end_to_end_run_directory():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split, run_dir, metrics = _tiny_run(tmp, two_heads=True)
        assert {p.name for p in run_dir.iterdir()} == set(schemas.RUN_FILES)
        card = json.loads((run_dir / "model_card.json").read_text())
        assert set(schemas.MODEL_CARD_KEYS) <= set(card) and len(card["classes"]) == 5
        preds = pq.read_table(run_dir / "preds.parquet")
        schemas.validate(preds, schemas.PREDS_SCHEMA)
        assert preds.num_rows == metrics["n_dev"] and set(preds.column("set").to_pylist()) == {"dev"}

        # The saved model reproduces the dev predictions it was selected on, on the same device it trained on.
        # On the GPU both runs use bf16, which keeps about 3 significant digits, so the tolerance is wider there.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, card = load_run(run_dir, device)
        dev = ViewDataset(split, ["dev"], classes=card["classes"], data_root=tmp / "store", views_root=tmp / "store")
        again = predict(model, make_loader(dev, 16, train=False, num_workers=0), GpuAugment(device), device)
        saved = preds.to_pandas().set_index("row_id")["p_fake"]
        rows = dev.rows["row_id"].to_numpy()[again["idx"]]
        atol = 1e-2 if device == "cuda" else 1e-5
        assert np.allclose(saved.loc[rows].to_numpy(), again["p_fake"], atol=atol)


def test_binary_only_run():
    with tempfile.TemporaryDirectory() as t:
        _, run_dir, _ = _tiny_run(Path(t), two_heads=False)
        card = json.loads((run_dir / "model_card.json").read_text())
        preds = pq.read_table(run_dir / "preds.parquet").to_pandas()
        assert card["classes"] == [] and preds["attr_probs"].map(len).eq(0).all()


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
