import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class FeatureBuilder:
    def __init__(self, metadata_path="data/processed/metadata.csv",
                 out_dir="data/processed/features"):
        self.metadata = pd.read_csv(metadata_path)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def load_csv_safe(self, path):
        if path is None or not isinstance(path, str) or not os.path.exists(path):
            return None
        return pd.read_csv(path).to_numpy()

    def build_all(self):
        print("Building feature tensors...")

        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            pid = row["participant_id"]
            out_path = os.path.join(self.out_dir, f"{pid}.npz")

            feats = {
                "covarep": self.load_csv_safe(row["covarep_path"]),
                "formant": self.load_csv_safe(row["formant_path"]),
                "au": self.load_csv_safe(row["au_path"]),
                "gaze": self.load_csv_safe(row["gaze_path"]),
                "pose": self.load_csv_safe(row["pose_path"]),
                "landmarks": self.load_csv_safe(row["landmarks_path"]),
                "label": row["phq8_binary"],
            }

            np.savez(out_path, **feats)

        print("Features saved → data/processed/features/")