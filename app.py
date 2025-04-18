import streamlit as st
import tensorflow as tf
import numpy as np
import os

st.title("ECG Arrhythmia Detection")

# Load model
model_path = "ecg_model_cpu.keras"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
else:
    st.error("Model file not found!")

# Load data
try:
    ecg_segments = np.load("processed_data/ecg_segments.npy")
    label_mapping = np.load("processed_data/label_mapping.npy", allow_pickle=True).item()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")

# Basic Prediction Interface (Demo)
uploaded_file = st.file_uploader("Upload ECG segment (.npy)", type="npy")

if uploaded_file is not None:
    try:
        sample = np.load(uploaded_file)
        prediction = model.predict(sample.reshape(1, -1, 1))
        predicted_class = np.argmax(prediction)
        st.write(f"Predicted class: {predicted_class}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
