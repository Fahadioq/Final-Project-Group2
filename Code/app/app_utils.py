"""Helpers shared by the app pages: run selection, mock labelling, the shared uploaded image,
model-input preprocessing (reused from etl.views, never reimplemented) and safe reading of result CSVs.

Run directories follow common.paths: <OF_DATA>/runs/<run_id>/ (OF_DATA defaults to ~/of_data).
OF_RUN_DIR (PLAN.md section 13) points the app at a single run instead.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:  # pages import common/ and etl/ from Code/
    sys.path.insert(0, str(CODE_DIR))

import streamlit as st  # noqa: E402

from common import paths  # noqa: E402
from etl import header_feats, views  # noqa: E402

EVAL_CROP = 224  # used when a model card has no eval_crop
MOCK_WARNING = "MOCK run: synthetic numbers for building and testing the app. These are not experiment results."
REQUIRED_COLUMNS = ("set", "group", "n_fake", "n_real", "small")
METRIC_COLUMNS = ("f1", "auroc", "tpr_at_5fpr", "balanced_acc", "oracle_f1")


# Runs

def runs_root() -> Path:
    """<OF_DATA>/runs, the same folder as common.paths.RUNS_DIR. OF_DATA is read on each call."""
    return Path(os.environ.get("OF_DATA", paths.OF_DATA)).expanduser() / "runs"


def list_runs(root: Path | None = None) -> list[Path]:
    """Run directories (folders with a model_card.json), sorted by name. OF_RUN_DIR selects one run."""
    single = os.environ.get("OF_RUN_DIR")
    if root is None and single:
        run = Path(single).expanduser()
        return [run] if (run / "model_card.json").is_file() else []
    root = Path(root or runs_root())
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "model_card.json").is_file())


def read_card(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "model_card.json").read_text())


def is_mock(run_dir: Path) -> bool:
    return bool(read_card(run_dir).get("mock")) or Path(run_dir).name.startswith("MOCK-")


def select_run() -> Path | None:
    """Sidebar run picker. The choice is kept in session state so every page uses the same run."""
    runs = list_runs()
    st.sidebar.caption(f"Runs: `{os.environ.get('OF_RUN_DIR') or runs_root()}`")
    if not runs:
        st.sidebar.info("No run directories found. Set OF_DATA (runs in <OF_DATA>/runs) or OF_RUN_DIR.")
        return None
    names = [r.name for r in runs]
    current = st.session_state.get("run_name")
    name = st.sidebar.selectbox("Run", names, index=names.index(current) if current in names else 0)
    st.session_state["run_name"] = name
    run_dir = runs[names.index(name)]
    if is_mock(run_dir):
        st.sidebar.warning("MOCK run (synthetic)")
    return run_dir


def mock_banner(run_dir: Path | None) -> None:
    if run_dir is not None and is_mock(run_dir):
        st.warning(MOCK_WARNING)


# The uploaded image, shared across pages

def image_input(page: str) -> tuple[bytes, str] | None:
    """Upload widget. The last uploaded file is kept in session state and reused on every page.

    Clear image bumps a counter that is part of every uploader key, so all uploaders start empty again
    and the cleared file cannot come back on a rerun.
    """
    gen = st.session_state.get("upload_gen", 0)
    key = f"upload_{page}_{gen}"
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], key=key)
    if up is not None:
        file_id = getattr(up, "file_id", up.name)
        if st.session_state.get(f"seen_{key}") != file_id:  # only a new file replaces the shared image
            st.session_state[f"seen_{key}"] = file_id
            st.session_state["image_bytes"] = up.getvalue()
            st.session_state["image_name"] = up.name
    if "image_bytes" not in st.session_state:
        return None
    left, right = st.columns([4, 1])
    left.caption(f"Current image: {st.session_state.get('image_name', 'uploaded image')}")
    if right.button("Clear image", key=f"clear_{page}"):
        st.session_state.pop("image_bytes", None)
        st.session_state.pop("image_name", None)
        st.session_state["upload_gen"] = gen + 1
        st.rerun()
    return st.session_state["image_bytes"], st.session_state.get("image_name", "uploaded image")


# Preprocessing (same recipe as evaluation)

def eval_crop(run_dir: Path | None) -> int:
    if run_dir is None:
        return EVAL_CROP
    return int(read_card(run_dir).get("eval_crop", EVAL_CROP))


def usable(data: bytes) -> tuple[bool, str]:
    """Whether the C view can be built without upscaling (same rule as the ETL)."""
    return views.eligibility(header_feats.header_features(data))


def model_input(img: Image.Image, crop: int = EVAL_CROP, factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """C view (or the downscaled C view when factor < 1), then the center crop the model sees.

    Returns (256 view, crop x crop model input), both uint8 HxWx3.
    """
    view = views.view_c(img) if factor >= 1.0 else views.view_c_scaled(img, factor)
    return view, views.center_crop(Image.fromarray(view), crop)


def jpeg(arr: np.ndarray, quality: int) -> np.ndarray:
    """JPEG round trip with Pillow. Evaluation uses GPU JPEG, so values can differ slightly."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def prediction_status(run_dir: Path | None) -> str:
    """Why no prediction is shown yet. Predictions wait for Code/inference.py (Modeling)."""
    if run_dir is None:
        return "No run selected."
    if not (Path(run_dir) / "model.pt").exists():
        return "This run has no checkpoint (model.pt), so it cannot score images."
    return "Prediction pending: the inference API in Code/inference.py is not integrated yet."


# Result CSVs

def truthy(col: pd.Series) -> pd.Series:
    """Booleans from a CSV column that may hold True/False, 1/0 or strings."""
    return col.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def read_csv_safely(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Read a CSV; returns (table, None) or (None, error message)."""
    try:
        return pd.read_csv(path), None
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
        return None, f"Could not read {Path(path).name}: {e}"


def has_mock_rows(df: pd.DataFrame) -> bool:
    return "mock" in df and bool(truthy(df["mock"]).any())


def check_metrics_table(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame | None, str | None]:
    """Check the columns of a per-group metrics table (analysis.metrics output) and coerce their types."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df]
    if missing:
        return None, f"{name} is missing columns: {missing}"
    metrics = [m for m in METRIC_COLUMNS if m in df]
    if not metrics:
        return None, f"{name} has no metric columns (expected one of {list(METRIC_COLUMNS)})"
    if df.empty:
        return None, f"{name} has no rows"
    df = df.copy()
    numeric = ["n_fake", "n_real", *metrics, *[f"{m}_{e}" for m in metrics for e in ("lo", "hi") if f"{m}_{e}" in df]]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # malformed values become NaN and show as n/a
    df["small"] = truthy(df["small"])
    df["set"], df["group"] = df["set"].astype(str), df["group"].astype(str)
    return df, None
