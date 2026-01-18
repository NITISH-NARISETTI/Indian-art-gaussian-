import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import pickle
import numpy as np
import cv2
from extract_features import extract
st.write("🚀 Streamlit app loaded")

st.set_page_config(page_title="Indian Art Gaussian Classifier")

st.title("🎨 Indian Art Style Classifier")
st.markdown("""
This app classifies Indian paintings using **Multivariate Gaussian models**
based on **color statistics (LAB space)**.
""")

@st.cache_resource
def load_model():
    with open("models/gaussians.pkl", "rb") as f:
        return pickle.load(f)

models = load_model()

uploaded = st.file_uploader("Upload a painting image", type=["jpg","png","jpeg"])

if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.read())

    x = extract("temp.jpg")

    scores = {k: float(v.logpdf(x)) for k,v in models.items()}
    pred = max(scores, key=scores.get)

    st.subheader(f"🧠 Predicted Style: **{pred}**")

    st.subheader("Log-Likelihoods")
    st.table(scores)
