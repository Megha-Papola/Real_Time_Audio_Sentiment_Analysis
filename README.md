#  Audio Emotion Recognition

A Streamlit-based web application that predicts the emotion expressed in a short speech audio clip using machine learning and deep audio features.

---

##  Features

*  Upload `.wav` audio files (≤ 3 seconds)
*  Listen to your uploaded audio clip
*  Predict emotions using a deep learning model
*  View prediction probabilities for all emotions
*  Real-time error handling for invalid or low-quality inputs

---

##  Project Structure

```
audio_emotion_recognition/
├── app.py                          # Streamlit web app
├── dataset/
│   └── features.csv                # Extracted features and labels
├── model/
│   ├── audio_sentiment_model.keras  # Trained Keras model
│   ├── label_encoder.pkl            # Label encoder
│   └── scaler.pkl                   # Feature scaler
├── utils/
│   └── featureExtraction.py        # Feature extraction module
├── audioFiles/                     # Raw audio files for training
├── extract_features_and_save.py    # Script to extract features
└── README.md
```

---

##  Technologies Used

| Tool/Library     | Purpose                       |
| ---------------- | ----------------------------- |
| `streamlit`      | Web application framework     |
| `librosa`        | Audio processing              |
| `tensorflow`     | Deep learning model           |
| `joblib`         | Save/load preprocessing tools |
| `sklearn`        | Scaling and encoding          |
| `numpy`/`pandas` | Data manipulation             |

---

##  Model Overview

* **Architecture**: Dense Neural Network

  * Layers: 512 -> 256 -> 128 -> 64 -> Output
  * Techniques: Dropout, BatchNormalization
* **Output**: Emotion classification via softmax
* **Training Enhancements**:

  * Early Stopping
  * ReduceLROnPlateau

---

##  Audio Features Extracted

| Feature Type       | Description                 |
| ------------------ | --------------------------- |
| Zero Crossing Rate | Measures noisiness/tonality |
| Chroma STFT        | Harmonic content            |
| MFCCs (20 coeffs)  | Timbre descriptors          |
| RMS Energy         | Energy of signal            |
| Mel Spectrogram    | Mel-scaled power spectrum   |
| Spectral Contrast  | Contrast in frequency bands |
| Spectral Bandwidth | Spread of spectrum          |
| Spectral Rolloff   | Energy rolloff point        |

---

##  Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/audio_emotion_recognition.git
cd audio_emotion_recognition
```

### 2. Create virtual environment

```bash
python -m venv ser_env
source ser_env\Scripts\activate on Windows  
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

##  Usage

1. Run the app.
2. Upload a `.wav` audio clip (3 seconds or less).
3. View the predicted emotion and confidence scores.

---

##  Future Enhancements

*  Support for mic recording
*  Multilingual emotion recognition
*  Deploy to cloud (e.g., Streamlit Cloud, Heroku)
*  Display waveform and spectrogram

---

##  Contributing

Feel free to fork the repository, raise issues, or submit pull requests!

---

##  Authors

* Megha Papola

---

