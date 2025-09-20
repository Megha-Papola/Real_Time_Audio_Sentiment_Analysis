import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import tempfile
from utils.featureExtraction import extract_features 


model = tf.keras.models.load_model("model/audio_sentiment_model.keras")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")


st.set_page_config(page_title="Audio Emotion Recognition", layout="centered")
st.title("🎙️ Audio Emotion Recognition")
st.markdown("Upload a short **WAV audio clip** and the model will predict the **emotion** being expressed.")


uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if uploaded_file is not None:
    try:
        st.audio(uploaded_file, format='audio/wav')

        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        
        features = extract_features(temp_path)

        if features is None or np.isnan(features).any():
            st.error("Failed to extract valid features. Please upload a clearer or slightly longer audio clip.")
        else:
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            
            st.subheader(" Predicted Emotion:")
            st.success(f"**{predicted_label}**")

            
            st.subheader(" Prediction Probabilities:")
            probs = {label: f"{prob*100:.2f}%" for label, prob in zip(label_encoder.classes_, prediction[0])}
            st.json(probs)

    except Exception as e:
        st.error(f" Error: {e}")
