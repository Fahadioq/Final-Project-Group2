"""Results: per-generator and per-lineage metrics.

Two separate sources: the selected run's analysis/ folder (written by analysis.metrics; mock runs live here)
and the repository's results/ folder (committed summaries of real runs only; mock rows are refused)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/app, for app_utils

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import app_utils as U  # noqa: E402
from common import paths  # noqa: E402

METRIC_LABELS = {"f1": "F1 at tau", "auroc": "AUROC", "tpr_at_5fpr": "TPR at 5% FPR",
                 "balanced_acc": "Balanced accuracy", "oracle_f1": "Oracle-tau F1"}


def show_group_table(df: pd.DataFrame) -> None:
    """Set and metric pickers, a bar chart, and a table with 95% CIs. df comes from U.check_metrics_table."""
    c1, c2 = st.columns(2)
    set_name = c1.selectbox("Evaluation set", sorted(df["set"].unique()))
    metric = c2.selectbox("Metric", [m for m in METRIC_LABELS if m in df], format_func=METRIC_LABELS.get)
    sub = df[df["set"] == set_name].sort_values("group")
    st.bar_chart(sub.set_index("group")[metric])
    table = sub[["group", "n_fake", "n_real", "small", metric]].copy()
    if f"{metric}_lo" in sub and f"{metric}_hi" in sub:
        table["95% CI"] = [f"{lo:.3f} to {hi:.3f}" if pd.notna(lo) and pd.notna(hi) else "n/a"
                           for lo, hi in zip(sub[f"{metric}_lo"], sub[f"{metric}_hi"])]
    st.dataframe(table, hide_index=True)
    if sub[metric].isna().any():
        st.caption("n/a = missing or malformed value in the CSV.")
    if sub["small"].any():
        st.caption("small = fewer than 30 fakes or reals in the balanced set; read with care.")


st.set_page_config(page_title="Results", layout="wide")
st.title("Results")
run_dir = U.select_run()
source = st.radio("Source", ["Selected run", "Repository results/"], horizontal=True)

if source == "Selected run":
    if run_dir is None:
        st.info("Select a run in the sidebar.")
        st.stop()
    U.mock_banner(run_dir)
    by = st.radio("Group by", ["generator", "lineage"], horizontal=True)
    path = run_dir / "analysis" / f"per_{by}.csv"
    if not path.exists():
        st.info(f"No {path.name} for this run yet. Generate it with:")
        st.code(f"cd Code && python -m analysis.metrics --run {run_dir} --index-root <index root>")
        st.stop()
    df, error = U.read_csv_safely(path)
    if error is None and U.has_mock_rows(df) and not U.is_mock(run_dir):
        error = "This file contains mock rows but the run is not a mock run. Regenerate it."
    if error is None:
        df, error = U.check_metrics_table(df, path.name)
    if error:
        st.error(error)
        st.stop()
    show_group_table(df)
else:
    files = sorted(paths.RESULTS_DIR.glob("*.csv"))
    if not files:
        st.info(f"No CSV summaries in {paths.RESULTS_DIR} yet.")
        st.stop()
    name = st.selectbox("Summary file", [f.name for f in files])
    df, error = U.read_csv_safely(paths.RESULTS_DIR / name)
    if error is None and U.has_mock_rows(df):
        error = "This file contains mock rows. Mock results do not belong in results/."
    if error:
        st.error(error)
        st.stop()
    st.caption(f"Committed summary: results/{name}")
    if set(U.REQUIRED_COLUMNS) <= set(df.columns):  # a per-group metrics table
        df, error = U.check_metrics_table(df, name)
        if error:
            st.error(error)
            st.stop()
        show_group_table(df)
    else:
        st.dataframe(df, hide_index=True)
