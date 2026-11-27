"""Checks for evaluate.py (no pytest, no weight download; uses the GPU when there is one).

    cd Code && python -m tests.test_evaluate

Trains a tiny ResNet-18 for two steps on a synthetic store with two test sets, then evaluates it.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from common import schemas
from data.standin import make_standin_store
from evaluate import evaluate, eval_sets_in
from train import DEFAULT_CONFIG, load_config, train

TEST_SETS = ("core_test", "reddit_test")


def _trained_run(tmp: Path) -> tuple[Path, Path, Path]:
    store = tmp / "store"
    split = make_standin_store(store, n_shards=2, rows_per_shard=60, seed=0, test_sets=TEST_SETS)
    cfg = load_config(DEFAULT_CONFIG, ["backbone=resnet18", "pretrained=false", "epochs=1", "batch_size=8",
                                       "eval_batch_size=16", "workers=0"])
    run_dir = tmp / "runs" / "r"
    train(cfg, split, run_dir, data_root=store, views_root=store, max_steps=2)
    return store, split, run_dir


def test_finds_test_sets():
    with tempfile.TemporaryDirectory() as t:
        split = make_standin_store(Path(t), n_shards=1, rows_per_shard=60, test_sets=TEST_SETS)
        assert eval_sets_in(split) == sorted(TEST_SETS)
        assert eval_sets_in(make_standin_store(Path(t) / "b", n_shards=1, rows_per_shard=60)) == []


def test_evaluate_writes_every_variant():
    with tempfile.TemporaryDirectory() as t:
        store, split, run_dir = _trained_run(Path(t))
        results = evaluate(run_dir, split, jpeg=[50], jpeg_fakes=[75], batch_size=16, workers=0,
                           data_root=store, views_root=store)
        expected = {f"{s}{v}" for s in TEST_SETS for v in ("", "@jpeg50", "@fakejpeg75")}
        assert set(results) == expected

        preds = pq.read_table(run_dir / "preds.parquet")
        schemas.validate(preds, schemas.PREDS_SCHEMA)
        df = preds.to_pandas()
        assert set(df["set"]) == expected | {"dev"}                   # dev rows from training are kept
        assert not df.duplicated(["set", "row_id"]).any()
        clean = df[df["set"] == "core_test"].set_index("row_id")["p_fake"]
        flipped = df[df["set"] == "core_test@fakejpeg75"].set_index("row_id")
        reals = flipped[flipped["label"] == 0]
        assert (clean.loc[reals.index] - reals["p_fake"]).abs().max() < 1e-2   # reals are untouched by the flip
        assert set(json.loads((run_dir / "metrics.json").read_text())["eval"]) == expected

        # Re-running replaces rows instead of adding a second copy.
        evaluate(run_dir, split, sets=["core_test"], batch_size=16, workers=0, data_root=store, views_root=store)
        again = pq.read_table(run_dir / "preds.parquet").to_pandas()
        assert len(again) == len(df) and not again.duplicated(["set", "row_id"]).any()


def test_unknown_set_is_rejected():
    with tempfile.TemporaryDirectory() as t:
        store, split, run_dir = _trained_run(Path(t))
        try:
            evaluate(run_dir, split, sets=["no_such_set"], workers=0, data_root=store, views_root=store)
        except ValueError as e:
            assert "no_such_set" in str(e)
        else:
            raise AssertionError("an unknown set name must be rejected")


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
