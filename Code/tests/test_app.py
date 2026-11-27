"""Smoke tests for the Streamlit app (no pytest, no model, no real data).

    cd Code && python -m tests.test_app

Every page runs through streamlit.testing.v1.AppTest against small temporary MOCK fixtures that follow
common.schemas.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image
from streamlit.testing.v1 import AppTest

from common import paths, schemas

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import app_utils as U  # noqa: E402

PAGES = {"detect": "pages/1_Detect.py", "shortcut": "pages/2_Shortcut_lens.py",
         "stress": "pages/3_Stress_test.py", "results": "pages/4_Results.py"}
RNG = np.random.default_rng(0)


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _write_fixture(root: Path) -> Path:
    """Two MOCK runs under <root>/runs (root plays OF_DATA): one with analysis CSVs, one without.
    Returns the runs folder."""
    classes = ["sdxl", "unknown-generator", "laion", "unknown-real"]
    for name, with_analysis in (("MOCK-fixture-s0", True), ("MOCK-noanalysis-s0", False)):
        run = root / "runs" / name
        run.mkdir(parents=True)
        card = {"run_id": name, "backbone": "resnet50", "classes": classes, "tau": 0.5,
                "view": "C", "eval_crop": 224, "mock": True}
        (run / "model_card.json").write_text(json.dumps(card))
        (run / "metrics.json").write_text(json.dumps({"run_id": name, "dev_auroc": 0.9, "tau": 0.5, "mock": True}))
        (run / "config.yaml").write_text(yaml.safe_dump({"mock": True}))
        attr = [[0.25] * 4, [0.25] * 4]
        preds = pa.table({"row_id": ["a", "b"], "set": ["core_test"] * 2, "label": np.array([1, 0], np.int8),
                          "p_fake": np.array([0.8, 0.2], np.float32), "attr_probs": attr,
                          "p_unknown_gen": np.array([0.25, 0.25], np.float32),
                          "p_unknown_real": np.array([0.25, 0.25], np.float32)}, schema=schemas.PREDS_SCHEMA)
        schemas.validate(preds, schemas.PREDS_SCHEMA)
        pq.write_table(preds, run / "preds.parquet")
        if with_analysis:
            (run / "analysis").mkdir()
            for by, groups in (("generator", ["sdxl", "flux.1-dev"]), ("lineage", ["sdxl", "flux1"])):
                rows = [{"run_id": name, "mock": True, "set": "core_test", "group_by": by, "group": g,
                         "n_fake": 100, "n_real": 100, "fakes_dropped": 0, "small": False,
                         **{m: v for m in ("f1", "auroc", "tpr_at_5fpr", "balanced_acc", "oracle_f1")},
                         **{f"{m}_{e}": v for m in ("f1", "auroc", "tpr_at_5fpr", "balanced_acc", "oracle_f1")
                            for e in ("lo", "hi")}}
                        for g, v in zip(groups, (0.9, 0.6))]
                pd.DataFrame(rows).to_csv(run / "analysis" / f"per_{by}.csv", index=False)
    return root / "runs"


def _app(page: str | None, runs_root: Path, **state) -> AppTest:
    """Run one page with OF_DATA set to the parent of runs_root, as common.paths expects."""
    os.environ["OF_DATA"] = str(runs_root.parent)
    os.environ.pop("OF_RUN_DIR", None)
    at = AppTest.from_file(str(APP / (PAGES[page] if page else "Home.py")), default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_list_runs_and_mock():
    with tempfile.TemporaryDirectory() as tmp:
        runs = _write_fixture(Path(tmp))
        found = U.list_runs(runs)
        assert [r.name for r in found] == ["MOCK-fixture-s0", "MOCK-noanalysis-s0"]
        assert all(U.is_mock(r) for r in found)
        assert U.list_runs(Path(tmp) / "missing") == []


def test_preprocessing_shapes():
    img = Image.open(io.BytesIO(_png(600, 400))).convert("RGB")
    view, x = U.model_input(img, 224)
    assert view.shape == (256, 256, 3) and x.shape == (224, 224, 3)
    view, x = U.model_input(img, 224, factor=0.7)
    assert x.shape == (224, 224, 3)
    assert U.jpeg(view, 50).shape == view.shape
    assert U.usable(_png(600, 400))[0] and not U.usable(_png(100, 100))[0]


def test_prediction_status_never_predicts():
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_fixture(Path(tmp)) / "MOCK-fixture-s0"
        assert "checkpoint" in U.prediction_status(run)
        assert U.prediction_status(None) == "No run selected."


def test_home():
    with tempfile.TemporaryDirectory() as tmp:
        at = _app(None, _write_fixture(Path(tmp)))
        assert "MOCK" in _texts(at.warning)
        empty = Path(tmp) / "empty" / "runs"
        empty.mkdir(parents=True)
        _app(None, empty)  # no runs: must still render


def test_pages_without_image():
    with tempfile.TemporaryDirectory() as tmp:
        runs = _write_fixture(Path(tmp))
        for page in ("detect", "shortcut", "stress"):
            assert "Upload" in _texts(_app(page, runs).info), page


def test_pages_with_shared_image():
    with tempfile.TemporaryDirectory() as tmp:
        runs = _write_fixture(Path(tmp))
        state = {"image_bytes": _png(600, 400), "image_name": "test.png"}
        for page in ("detect", "stress"):
            info = _texts(_app(page, runs, **state).info)
            assert "checkpoint" in info, page  # pending message, no prediction shown
        assert len(_app("shortcut", runs, **state).dataframe) >= 1


def test_detect_rejects_small_image():
    with tempfile.TemporaryDirectory() as tmp:
        at = _app("detect", _write_fixture(Path(tmp)), image_bytes=_png(100, 100))
        assert "cannot be scored" in _texts(at.error)


def test_results_page():
    with tempfile.TemporaryDirectory() as tmp:
        runs = _write_fixture(Path(tmp))
        at = _app("results", runs, run_name="MOCK-fixture-s0")
        assert "MOCK" in _texts(at.warning) and len(at.dataframe) == 1
        at = _app("results", runs, run_name="MOCK-noanalysis-s0")
        assert "No per_generator.csv" in _texts(at.info)


def test_run_paths_follow_common_paths():
    with tempfile.TemporaryDirectory() as tmp:
        runs = _write_fixture(Path(tmp))
        os.environ["OF_DATA"] = tmp
        os.environ.pop("OF_RUN_DIR", None)
        assert U.runs_root() == Path(tmp) / "runs" and len(U.list_runs()) == 2
        os.environ["OF_RUN_DIR"] = str(runs / "MOCK-noanalysis-s0")  # PLAN section 13: one run
        try:
            assert [r.name for r in U.list_runs()] == ["MOCK-noanalysis-s0"]
        finally:
            os.environ.pop("OF_RUN_DIR")


def test_clear_image_resets_uploader():
    with tempfile.TemporaryDirectory() as tmp:
        at = _app("detect", _write_fixture(Path(tmp)), image_bytes=_png(600, 400), image_name="a.png")
        at.button(key="clear_detect").click().run()
        assert not at.exception, at.exception
        assert "image_bytes" not in at.session_state
        assert at.session_state["upload_gen"] == 1  # every uploader gets a new, empty key
        assert "Upload" in _texts(at.info)


def test_read_and_check_csv():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "empty.csv").write_text("")
        (d / "binary.csv").write_bytes(b"\xff\xfe\x00garbage\x00\xff")
        (d / "missing.csv").write_text("set,group\ncore_test,sdxl\n")
        (d / "bad_values.csv").write_text("set,group,n_fake,n_real,small,auroc,auroc_lo,auroc_hi\n"
                                          "core_test,sdxl,100,100,False,oops,0.8,0.9\n")
        assert U.read_csv_safely(d / "empty.csv")[1] is not None
        df, err = U.read_csv_safely(d / "binary.csv")
        assert err is not None or U.check_metrics_table(df, "binary.csv")[1] is not None
        df, _ = U.read_csv_safely(d / "missing.csv")
        assert "missing columns" in U.check_metrics_table(df, "missing.csv")[1]
        df, _ = U.read_csv_safely(d / "bad_values.csv")
        ok, err = U.check_metrics_table(df, "bad_values.csv")
        assert err is None and pd.isna(ok["auroc"].iloc[0]) and not ok["small"].iloc[0]


def test_results_page_malformed_csv():
    with tempfile.TemporaryDirectory() as tmp:
        runs = _write_fixture(Path(tmp))
        (runs / "MOCK-fixture-s0" / "analysis" / "per_generator.csv").write_text("set,group\ncore_test,sdxl\n")
        at = _app("results", runs, run_name="MOCK-fixture-s0")
        assert "missing columns" in _texts(at.error)


def _results_source(runs: Path, results_dir: Path) -> AppTest:
    """Results page on the 'Repository results/' source, with results/ redirected to results_dir."""
    old = paths.RESULTS_DIR
    paths.RESULTS_DIR = results_dir  # the page reads common.paths at run time
    try:
        at = _app("results", runs)
        at.radio[0].set_value("Repository results/").run()
        assert not at.exception, at.exception
        return at
    finally:
        paths.RESULTS_DIR = old


def test_repository_results_rejects_mock():
    with tempfile.TemporaryDirectory() as tmp:
        runs = _write_fixture(Path(tmp))
        results = Path(tmp) / "results"
        results.mkdir()
        mock_csv = runs / "MOCK-fixture-s0" / "analysis" / "per_generator.csv"
        (results / "per_generator.csv").write_text(mock_csv.read_text())
        at = _results_source(runs, results)
        assert "Mock results do not belong" in _texts(at.error)

        real = pd.read_csv(mock_csv).drop(columns="mock")  # same table without mock rows
        real.to_csv(results / "per_generator.csv", index=False)
        at = _results_source(runs, results)
        assert not at.error and len(at.dataframe) == 1


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
