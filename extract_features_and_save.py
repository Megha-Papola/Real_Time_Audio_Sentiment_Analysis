import os
import pandas as pd
import numpy as np
from utils.featureExtraction import extract_features  


metadata_path = os.path.join("dataset", "metadata.csv")
df = pd.read_csv(metadata_path)


features_list = []
emotions = []

for idx, row in df.iterrows():
    file_path = row["path"]
    emotion = row["emotion"]

    try:
        features = extract_features(file_path)  
        if features is not None:
            features_list.append(features)
            emotions.append(emotion)
        else:
            print(f"Skipping {file_path} due to feature extraction failure.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


features_array = np.array(features_list)
features_df = pd.DataFrame(features_array)
features_df['emotion'] = emotions


output_path = "dataset/features.csv"
features_df.to_csv(output_path, index=False)
print(f"Features saved to {output_path} with correct format.")