"""Streamlit app entry point.  Run: streamlit run Code/app/Home.py

Owner: Analysis & App. Pages to add under Code/app/pages/: Detect, Shortcut lens, Stress test, Results.
The app imports etl.views / etl.header_feats (Data) and inference.load (Modeling); it never reimplements
preprocessing.
"""
import streamlit as st

st.set_page_config(page_title="OpenFake generalization", layout="wide")
st.title("Cross-generator generalization on OpenFake")
st.write("Placeholder. Pages: Detect, Shortcut lens, Stress test, Results.")
