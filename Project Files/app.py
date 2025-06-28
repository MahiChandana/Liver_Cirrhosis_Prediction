import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and preprocessing tools
with open("best_model.pkl", "rb") as file:
    model, scaler, label_encoders, feature_names = pickle.load(file)

# Page settings
st.set_page_config(page_title="Liver Cirrhosis Detection", page_icon="🩺", layout="centered")

# Sidebar branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4743/4743036.png", width=100)
    st.title("Liver Cirrhosis 🩺")
    st.caption("By Your ML Team")
    st.markdown("---")

# Main title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Liver Cirrhosis Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the patient details below to predict liver cirrhosis risk.</p>", unsafe_allow_html=True)
st.markdown("")

# Normalize label_encoder keys
normalized_encoders = {key.lower(): value for key, value in label_encoders.items()}

# Split form into two columns for better layout
col1, col2 = st.columns(2)
user_input = {}

# Arrange features in two-column layout
for idx, feature in enumerate(feature_names):
    feature_lower = feature.lower()

    with col1 if idx % 2 == 0 else col2:
        # Dropdown for categorical fields
        if feature_lower in normalized_encoders:
            le = normalized_encoders[feature_lower]
            options = list(le.classes_)
            selection = st.selectbox(f"{feature} 🧬", options)
            user_input[feature] = le.transform([selection])[0]

        # Age-specific range
        elif "age" in feature_lower:
            user_input[feature] = st.number_input(f"{feature} 🎂", min_value=1, max_value=120, step=1)

        # Other numeric features
        else:
            user_input[feature] = st.number_input(f"{feature} 🔢", format="%.2f")

# Prediction Button
st.markdown("----")
if st.button("🔍 Predict Now"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = "🛑 High Risk of Cirrhosis" if prediction == 1 else "✅ Likely Healthy"

    # Color-coded output
    st.markdown(f"<div style='background-color: {'#FFCCCC' if prediction == 1 else '#D4EDDA'}; \
                 padding: 15px; border-radius: 10px; text-align: center; font-size: 20px;'>\
                 <strong>{result}</strong></div>", unsafe_allow_html=True)
