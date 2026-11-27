"""Checks for analysis/make_mock_run.py and analysis/metrics.py (no pytest, no GPU, no real data).

    cd Code && python -m tests.test_analysis

All runs here are MOCK runs written to a temporary folder; nothing is written into the repo.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis import make_mock_run as mock
from analysis import metrics as M
from common import paths, schemas

RNG = np.random.default_rng(0)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


def test_metrics_hand_example():
    # sklearn's documented example: AUROC 0.75
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    s = M.score(y, p, tau=0.5)
    assert _close(s["auroc"], 0.75)
    assert _close(s["f1"], 2 / 3)              # predicts only 0.8 as fake: TP 1, FN 1, FP 0
    assert _close(s["balanced_acc"], 0.75)     # TPR 0.5, TNR 1.0
    assert _close(s["oracle_f1"], 0.8)         # tau 0.35: TP 2, FP 1
    assert _close(s["tpr_at_5fpr"], 0.5)       # no false positives allowed: only 0.8 is caught


def test_metrics_perfect_reversed_and_tied():
    y = [0, 0, 1, 1]
    perfect = M.score(y, [0.1, 0.2, 0.8, 0.9], tau=0.5)
    assert all(_close(perfect[m], 1.0) for m in M.METRICS)
    assert _close(M.score(y, [0.9, 0.8, 0.2, 0.1], tau=0.5)["auroc"], 0.0)
    tied = M.score(y, [0.5, 0.5, 0.5, 0.5], tau=0.5)  # every image predicted fake
    assert _close(tied["auroc"], 0.5)
    assert _close(tied["tpr_at_5fpr"], 0.0)
    assert _close(tied["balanced_acc"], 0.5)
    assert _close(tied["oracle_f1"], 2 / 3)


def test_missing_class_gives_nan():
    for y in ([1, 1, 1], [0, 0], []):
        s = M.score(y, [0.5] * len(y), tau=0.5)
        assert all(math.isnan(v) for v in s.values())
        ci = M.bootstrap_ci(y, [0.5] * len(y), tau=0.5, n_boot=10)
        assert all(math.isnan(lo) and math.isnan(hi) for lo, hi in ci.values())


def test_oracle_f1_at_least_fixed_f1():
    for _ in range(20):
        y = RNG.integers(0, 2, 200)
        y[:2] = [0, 1]
        p = np.round(RNG.random(200), 2)
        for tau in (0.1, 0.5, 0.9):
            s = M.score(y, p, tau)
            assert s["oracle_f1"] >= s["f1"] - 1e-12


def test_bootstrap_deterministic_and_stratified():
    y = np.r_[np.zeros(40, int), np.ones(25, int)]
    p = np.clip(y * 0.4 + RNG.random(65) * 0.6, 0, 1)
    a = M.bootstrap_ci(y, p, tau=0.5, n_boot=50, seed=3)
    b = M.bootstrap_ci(y, p, tau=0.5, n_boot=50, seed=3)
    assert a == b
    assert all(lo <= hi for lo, hi in a.values())
    idx = M.stratified_resample(y, np.random.default_rng(0))
    assert int(y[idx].sum()) == 25 and len(idx) == 65  # class counts kept in every resample


def test_balanced_set():
    df = pd.DataFrame({"label": [1] * 10 + [1] * 5 + [0] * 30,
                       "generator": ["a"] * 10 + ["b"] * 5 + ["laion"] * 30})
    bal, dropped = M.balanced_set(df, "a")
    assert (bal["label"] == 1).sum() == (bal["label"] == 0).sum() == 10 and dropped == 0
    assert set(bal.loc[bal["label"] == 1, "generator"]) == {"a"}
    again, _ = M.balanced_set(df, "a")
    assert again.index.equals(bal.index)                       # same seed and group -> same sample
    few_reals = pd.concat([df[df["label"] == 1], df[df["label"] == 0].iloc[:3]])   # 3 reals only
    bal, dropped = M.balanced_set(few_reals, "a")
    assert len(bal) == 6 and dropped == 7


def test_attach_index_checks():
    index = pd.DataFrame({"row_id": ["r1", "r2", "r3"], "pool": "p", "label": ["fake", "real", "fake"],
                          "model": ["sd-1.5", "laion", "stable-diffusion-v1-5"], "lineage": ["sd12", "real", "sd12"]})
    preds = pd.DataFrame({"row_id": ["r1", "r2", "r3"], "set": "s", "label": np.array([1, 0, 1], np.int8),
                          "p_fake": [0.9, 0.1, 0.8]})
    df = M.attach_index(preds, index, group_map={"stable-diffusion-v1-5": "sd-1.5"})
    assert list(df["generator"]) == ["sd-1.5", "laion", "sd-1.5"]   # canonical name merges the duplicate

    for bad, why in [
        (pd.concat([preds, preds.iloc[:1]]), "duplicate"),
        (preds.assign(row_id=["r1", "r2", "zz"]), "not in the index"),
        (preds.assign(label=np.array([0, 0, 1], np.int8)), "disagrees"),
    ]:
        try:
            M.attach_index(bad, index)
        except ValueError as e:
            assert why in str(e), e
        else:
            raise AssertionError(f"expected an error for: {why}")


def test_mock_run_files_and_schemas():
    with tempfile.TemporaryDirectory() as tmp:
        run_dirs = mock.write_mock(tmp, seeds=(0,))
        for pool in set(mock.SETS.values()):
            schemas.validate(pq.read_table(paths.index_path(pool, mock.CHUNK, root=tmp)), schemas.INDEX_SCHEMA)
        index = M.read_index(tmp)
        for run_dir in run_dirs:
            assert run_dir.name.startswith("MOCK-")
            assert not (run_dir / "model.pt").exists()
            run = M.read_run(run_dir)
            assert run["mock"] and not run["has_checkpoint"]
            assert run["preds"]["row_id"].isin(index["row_id"]).all()
            n_classes = len(run["card"]["classes"])
            assert all(len(a) == n_classes for a in run["preds"]["attr_probs"])
            assert set(run["preds"]["set"]) == set(mock.SETS)


def test_mock_run_reproducible():
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        a = mock.write_mock(t1, seeds=(0,))[0]
        b = mock.write_mock(t2, seeds=(0,))[0]
        assert pq.read_table(a / "preds.parquet").equals(pq.read_table(b / "preds.parquet"))
        assert (a / "model_card.json").read_text() == (b / "model_card.json").read_text()


def test_read_run_requires_files():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = mock.write_mock(tmp, seeds=(0,))[0]
        (run_dir / "metrics.json").unlink()
        try:
            M.read_run(run_dir)
        except FileNotFoundError as e:
            assert "metrics.json" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")


def test_end_to_end_on_mock():
    with tempfile.TemporaryDirectory() as tmp:
        run_dirs = mock.write_mock(tmp, seeds=(0, 1))
        index = M.read_index(tmp)
        tables = {}
        for run_dir in run_dirs:
            run = M.read_run(run_dir)
            df = M.attach_index(run["preds"], index)
            tables[run["run_id"]] = M.evaluate_groups(df, run["card"]["tau"], by="lineage",
                                                      sets=["core_test"], n_boot=20)
            gens = M.evaluate_groups(df, run["card"]["tau"], by="generator", sets=["core_test"], n_boot=5)
            assert set(gens["group"]) == set(mock.GENERATORS)
            assert (gens["n_fake"] == gens["n_real"]).all()

        # The mock makes held-out flux1 harder; this checks the plumbing, not a real finding.
        lolo = tables["MOCK-e2-holdout-flux1-s0"].set_index("group")
        assert lolo.loc["flux1", "auroc"] < lolo.loc["sdxl", "auroc"]

        e0 = tables["MOCK-e0-all-s0"]
        per_group = M.generalization_gap(tables["MOCK-e2-holdout-flux1-s0"], e0)
        assert per_group.set_index("group").loc["flux1", "gap_auroc"] > 0
        run = M.read_run(run_dirs[1])
        ref = M.pooled_reference(M.attach_index(run["preds"], index), run["card"]["tau"], "core_validation",
                                 groups=run["config"]["train_lineages"])
        scalar = M.generalization_gap(tables["MOCK-e2-holdout-flux1-s0"], ref)
        assert "gap_f1" in scalar and "gap_auroc" in scalar

        agg = M.mean_sd([tables["MOCK-e0-all-s0"], tables["MOCK-e0-all-s1"]], seeds=[0, 1])
        assert (agg["n_seeds"] == 2).all() and "auroc_sd" in agg


def _raises(fn, exc=ValueError) -> str:
    try:
        fn()
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def test_score_rejects_bad_inputs():
    y = [0, 1]
    _raises(lambda: M.score([0, 2], [0.1, 0.9], 0.5))            # label not 0/1
    _raises(lambda: M.score(y, [0.1, 1.5], 0.5))                 # probability above 1
    _raises(lambda: M.score(y, [0.1, float("nan")], 0.5))        # NaN score
    _raises(lambda: M.score(y, [0.1, 0.9, 0.5], 0.5))            # shape mismatch
    for tau in (float("nan"), -0.1, 1.5):
        _raises(lambda: M.score(y, [0.1, 0.9], tau))


def test_bootstrap_rejects_bad_args():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    for n_boot in (0, -5, 2.5):
        _raises(lambda: M.bootstrap_ci(y, p, 0.5, n_boot=n_boot))
    for level in (0.0, 1.0, 1.2):
        _raises(lambda: M.bootstrap_ci(y, p, 0.5, n_boot=5, level=level))


def test_balanced_set_ignores_row_order():
    n = 60
    df = pd.DataFrame({"row_id": [f"r{i:03d}" for i in range(n)],
                       "label": [1] * 20 + [0] * 40,
                       "generator": ["a"] * 20 + ["laion"] * 40})
    a, _ = M.balanced_set(df, "a")
    b, _ = M.balanced_set(df.sample(frac=1, random_state=7), "a")
    assert sorted(a["row_id"]) == sorted(b["row_id"])


def test_read_run_rejects_bad_preds():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = mock.write_mock(tmp, seeds=(0,))[0]
        table = pq.read_table(run_dir / "preds.parquet")
        p = table.column("p_fake").to_numpy().copy()
        p[0] = 1.5
        bad = table.set_column(table.schema.get_field_index("p_fake"), "p_fake", pa.array(p, pa.float32()))
        pq.write_table(bad, run_dir / "preds.parquet")
        assert "[0, 1]" in _raises(lambda: M.read_run(run_dir))


def test_mean_sd_rejects_duplicate_seeds():
    t = pd.DataFrame({"set": ["s"], "group_by": ["lineage"], "group": ["a"],
                      **{m: [0.5] for m in M.METRICS}})
    assert "duplicate seeds" in _raises(lambda: M.mean_sd([t, t], seeds=[0, 0]))
    _raises(lambda: M.mean_sd([t, t], seeds=[0]))
    _raises(lambda: M.mean_sd([pd.concat([t, t])], seeds=[0]))  # repeated rows inside one table
    ok = M.mean_sd([t, t.assign(auroc=0.7)], seeds=[0, 1])
    assert ok["n_seeds"].iloc[0] == 2 and _close(ok["auroc_mean"].iloc[0], 0.6)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
