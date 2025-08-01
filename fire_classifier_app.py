import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="Fire Type Classifier", layout="centered")

# Custom CSS for purple theme
st.markdown("""
    <style>
    body {
        background-color: #f7f3fa;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
        color: #5a287d;
    }
    .stTitle {
        color: #5a287d;
    }
    h1, h2, h3 {
        color: #5a287d;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton button {
        background-color: #a463c2;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #833bb5;
    }
    .stNumberInput, .stSelectbox {
        border-color: #b28ac6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title(" Fire Type Classification")
st.markdown("Use **MODIS satellite readings** to predict the type of fire with gentle confidence. ")

# Divider
st.markdown("---")

# Input fields
brightness = st.number_input(" Brightness", value=300.0)
bright_t31 = st.number_input(" Brightness T31", value=290.0)
frp = st.number_input(" Fire Radiative Power (FRP)", value=15.0)
scan = st.number_input(" Scan", value=1.0)
track = st.number_input(" Track", value=1.0)
confidence = st.selectbox(" Confidence Level", ["low", "nominal", "high"])

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Prepare input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

# Prediction
if st.button(" Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: " Vegetation Fire",
        2: " Other Static Land Source",
        3: "Offshore Fire"
    }

    result = fire_types.get(prediction, "Unknown")
    st.success(f"**Predicted Fire Type:** {result}")

# Footer
st.markdown("---")
