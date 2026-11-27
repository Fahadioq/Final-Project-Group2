"""Streamlit app entry point.  Run from the repo root: streamlit run Code/app/Home.py

Pages (Code/app/pages/): Detect, Shortcut lens, Stress test, Results. The app reuses etl.views and
etl.header_feats for preprocessing and will use Code/inference.py for predictions.
"""
import json

import streamlit as st

import app_utils as U

st.set_page_config(page_title="OpenFake generalization", layout="wide")
st.title("Cross-generator generalization on OpenFake")
st.write("Explore how a detector trained on some image generators handles generators it has never seen.")

run_dir = U.select_run()
U.mock_banner(run_dir)

st.subheader("Pages")
st.markdown(
    "- **Detect**: upload an image and see exactly what the model receives.\n"
    "- **Shortcut lens**: the file's metadata (format, size, JPEG quality), which detectors can exploit.\n"
    "- **Stress test**: JPEG compression and downscaling applied to the uploaded image.\n"
    "- **Results**: per-generator and per-lineage metrics for the selected run."
)
st.info("Week 1 skeleton: model predictions appear once the inference API is available.")

if run_dir is not None:
    st.subheader("Selected run")
    card = U.read_card(run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backbone", card.get("backbone", "?"))
    c2.metric("Threshold tau", f"{card['tau']:.3f}" if "tau" in card else "?")
    c3.metric("View / crop", f"{card.get('view', '?')} / {card.get('eval_crop', '?')}")
    c4.metric("Checkpoint", "yes" if (run_dir / "model.pt").exists() else "no")
    auroc = metrics.get("dev_auroc")
    st.caption(f"`{run_dir}` · dev AUROC " + (f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "n/a"))
