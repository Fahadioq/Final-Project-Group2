"""Detect: upload an image and see the exact model input (C view, then the evaluation center crop).
Predictions are shown once Code/inference.py is available."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/app, for app_utils

import streamlit as st  # noqa: E402

import app_utils as U  # noqa: E402
from etl import views  # noqa: E402

st.set_page_config(page_title="Detect", layout="wide")
st.title("Detect")
run_dir = U.select_run()
U.mock_banner(run_dir)

image = U.image_input("detect")
if image is None:
    st.info("Upload an image to see what the model receives.")
    st.stop()

data, _ = image
ok, reason = U.usable(data)
if not ok:
    st.error(f"This image cannot be scored ({reason}): the C view needs a short side of at least 256 px "
             "without upscaling, the same rule the dataset pipeline uses.")
    st.stop()

img = views.load_rgb(data)
crop = U.eval_crop(run_dir)
view, x = U.model_input(img, crop)

c1, c2, c3 = st.columns(3)
c1.image(img, caption=f"Uploaded ({img.width} x {img.height})", width=320)
c2.image(view, caption="C view (256 x 256)", width=256)
c3.image(x, caption=f"Model input: center {crop} x {crop}", width=crop)

st.subheader("Prediction")
st.info(U.prediction_status(run_dir))
