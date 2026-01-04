import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Earnings Manipulator Detection")

st.title("📊 Earnings Manipulator Detection App")

# ----------------------------
# Debug: show current directory
# ----------------------------
st.write("📁 Current working directory:")
st.code(os.getcwd())

st.write("📄 Files available in directory:")
st.code(os.listdir(os.getcwd()))

# ----------------------------
# Load model and features safely
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

@st.cache_resource
def load_files():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_PATH, "rb") as f:
        features = pickle.load(f)
    return model, features

try:
    model, features = load_files()
except Exception as e:
    st.error("❌ Model files not found or could not be loaded.")
    st.exception(e)   # THIS WILL SHOW THE REAL ERROR
    st.stop()

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("🔢 Financial Inputs")

inputs = []
for feature in features:
    value = st.number_input(
        label=str(feature),
        value=0.0,
        format="%.4f"
    )
    inputs.append(value)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely Earnings Manipulator (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Not an Earnings Manipulator (Confidence: {(1 - probability):.2%})")
