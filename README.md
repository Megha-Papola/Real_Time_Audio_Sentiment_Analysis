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
## Dataset
This project uses a combination of publicly available emotional speech datasets to train and evaluate the model:

* **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset):** Contains 7,442 audio clips from 91 actors. Emotions include anger, disgust, fear, happiness, neutral, and sadness. The recordings vary in sentence structure and intensity, providing diversity in speech patterns and emotional expression.

* **TESS (Toronto Emotional Speech Set):** Comprises 2,800 recordings spoken by 2 female actors, portraying seven emotions — happy, sad, angry, fear, disgust, pleasant surprise, and neutral. It is widely used for benchmarking emotional speech models.

* **SAVEE (Surrey Audio-Visual Expressed Emotion):** Includes 480 samples from 4 male speakers, each expressing 7 distinct emotions. This dataset adds variety in gender and expression.

* **Custom Preprocessed Dataset:** Features from the above datasets were extracted and stored in features.csv, including Zero Crossing Rate, MFCCs, Chroma, RMS Energy, and more. This processed data is used for training and inference within the app.


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
* Tanuja Bisht
* Purvi Joshi
* Esha Danu

---

