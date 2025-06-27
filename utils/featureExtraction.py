import os
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm



def extract_features(file_path):
    try:
        # Load audio inside the function
        data, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

        n_fft = 512
        hop_length = 256
        stft = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))

        features = []

        features.append(np.mean(librosa.feature.zero_crossing_rate(data, frame_length=n_fft, hop_length=hop_length)))

        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        features.extend(np.mean(chroma, axis=1))

        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
        features.extend(np.mean(mfcc, axis=1))

        rms = librosa.feature.rms(y=data, frame_length=n_fft, hop_length=hop_length)
        features.append(np.mean(rms))

        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        features.extend(np.mean(mel, axis=1))

        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        features.extend(np.mean(contrast, axis=1))

        bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        features.append(np.mean(bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        features.append(np.mean(rolloff))

        return np.array(features)

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

    
def get_emotion_label(file_path):
    file_name = os.path.basename(file_path).lower()

    # TESS
    if "tess" in file_path.lower():
        parts = file_name.split("_")
        if len(parts) >= 3:
            return parts[-1].replace(".wav", "")  # e.g., "happy"

    # RAVDESS
    elif "ravdess" in file_path.lower():
        return file_name.split("-")[2]  # e.g., '03'

    # CREMA-D
    elif "crema" in file_path.lower():
        return file_name.split("_")[-1].replace(".wav", "")  # e.g., "ANG"

    # SAVEE
    elif "savee" in file_path.lower():
        return file_name[-6:-4]  # e.g., "sa" for sad

    else:
        return "unknown"

def process_directory_parallel(data_dir, n_jobs=-1):
    audio_files = [
        os.path.join(dp, f)
        for dp, _, filenames in os.walk(data_dir)
        for f in filenames if f.endswith('.wav')
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_features)(f) for f in tqdm(audio_files, desc="Extracting Features")
    )

    return [r for r in results if r is not None]

def save_to_csv(features_with_labels, output_path="features.csv"):
    df = pd.DataFrame(features_with_labels)
    n_features = df.shape[1] - 1
    df.columns = [str(i) for i in range(1, n_features + 1)] + ['emotion']
    df.to_csv(output_path, index=False)
    print(f"✅ Features saved to {output_path}")

if __name__ == "__main__":
    data_dir = "audioFiles"  # replace with your path
    features_with_labels = process_directory_parallel(data_dir)
    save_to_csv(features_with_labels, output_path="dataset/features.csv")
