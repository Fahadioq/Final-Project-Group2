"""Stress test: apply JPEG compression and downscaling to the uploaded image and preview the model input.
Live re-prediction is added once Code/inference.py is available."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/app, for app_utils

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

import app_utils as U  # noqa: E402
from etl import views  # noqa: E402

st.set_page_config(page_title="Stress test", layout="wide")
st.title("Stress test")
run_dir = U.select_run()
U.mock_banner(run_dir)

image = U.image_input("stress")
if image is None:
    st.info("Upload an image to stress-test it.")
    st.stop()

data, _ = image
ok, reason = U.usable(data)
if not ok:
    st.error(f"This image cannot be scored ({reason}).")
    st.stop()

img = views.load_rgb(data)
crop = U.eval_crop(run_dir)
c1, c2 = st.columns(2)
factor = c1.slider("Downscale factor", 0.25, 1.0, 1.0, 0.05)
use_jpeg = c2.checkbox("Apply JPEG compression", value=False)
quality = c2.slider("JPEG quality", 10, 100, 75, 5, disabled=not use_jpeg)

_, baseline = U.model_input(img, crop)
try:
    view, _ = U.model_input(img, crop, factor=factor)
except ValueError:
    st.error(f"At factor {factor:.2f} the image is smaller than the 256 px view. Use a larger factor.")
    st.stop()
if use_jpeg:
    view = U.jpeg(view, quality)  # JPEG on the 256 view, before the crop (as in the robustness tests)
stressed = views.center_crop(Image.fromarray(view), crop)

c1, c2 = st.columns(2)
c1.image(baseline, caption="Model input, unchanged", width=crop)
label = f"downscale x{factor:.2f}" + (f", JPEG q{quality}" if use_jpeg else "")
c2.image(stressed, caption=f"Model input, {label}", width=crop)
st.caption("Preview uses Pillow JPEG; the evaluation uses GPU JPEG, so pixels can differ slightly.")

st.subheader("Prediction")
st.info(U.prediction_status(run_dir))
