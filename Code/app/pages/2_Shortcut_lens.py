"""Shortcut lens: the uploaded file's header features, which a detector could exploit without looking at
pixels. The metadata-only classifier's verdict is added in a later step."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/app, for app_utils

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import app_utils as U  # noqa: E402
from common import paths  # noqa: E402
from etl.header_feats import header_features  # noqa: E402

st.set_page_config(page_title="Shortcut lens", layout="wide")
st.title("Shortcut lens")
run_dir = U.select_run()
U.mock_banner(run_dir)

image = U.image_input("shortcut")
if image is None:
    st.info("Upload an image to inspect its file metadata.")
    st.stop()

data, _ = image
hf = header_features(data)
if not hf["header_ok"]:
    st.error("Could not read this file's header.")
    st.stop()

w, h = hf["width"], hf["height"]
rows = {
    "Format": hf["fmt"],
    "Width x height": f"{w} x {h}",
    "Megapixels": f"{w * h / 1e6:.2f}",
    "Width and height divisible by 64": "yes" if w % 64 == 0 and h % 64 == 0 else "no",
    "File size (KB)": f"{hf['file_bytes'] / 1024:.0f}",
    "Bits per pixel": f"{hf['file_bytes'] * 8 / (w * h):.2f}",
    "JPEG quality (estimate)": hf["jpeg_quality"] if hf["jpeg_quality"] is not None else "n/a",
    "Chroma subsampling": hf["jpeg_subsampling"] or "n/a",
    "EXIF present": "yes" if hf["has_exif"] else "no",
    "ICC profile present": "yes" if hf["has_icc"] else "no",
}
st.dataframe(pd.DataFrame({"Feature": list(rows), "Value": [str(v) for v in rows.values()]}),
             hide_index=True)

st.subheader("Metadata-only classifier")
st.info("Verdict pending: the metadata-only baseline (E3.1) is not built yet.")

summary = paths.RESULTS_DIR / "header_summary_core_train.csv"
if summary.exists():
    with st.expander("For comparison: header statistics of the training split (core_train)"):
        st.dataframe(pd.read_csv(summary), hide_index=True)
