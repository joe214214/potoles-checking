"""
Terminal server (Base Station).

Receives fused feature vectors from Cluster Heads and runs the
pre-trained Random Forest model for final pothole classification.
"""

import numpy as np
import joblib


class BaseStation:
    def __init__(self, model_path):
        raw = joblib.load(model_path)
        # train_model.py saves the model directly; how_to_load.py wraps in dict.
        # Handle both cases.
        self.model = raw["model"] if isinstance(raw, dict) else raw
        self.predictions = []   # list of result dicts

    def receive_and_predict(self, fused_data):
        """
        fused_data: dict with keys
          features (list[11]), lat, lon, timestamp, cluster_id, num_sensors
        Returns the same dict augmented with 'prediction' (0 or 1).
        """
        features = np.array(fused_data["features"], dtype=float).reshape(1, -1)
        prediction = int(self.model.predict(features)[0])

        result = {
            "timestamp":   fused_data["timestamp"],
            "lat":         fused_data["lat"],
            "lon":         fused_data["lon"],
            "prediction":  prediction,
            "cluster_id":  fused_data["cluster_id"],
            "num_sensors": fused_data["num_sensors"],
        }
        self.predictions.append(result)
        return result
