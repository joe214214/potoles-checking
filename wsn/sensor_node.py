"""
Vehicle sensor node.

Each node loads one trip-vehicle CSV file. It slides a window of
WINDOW_SIZE readings, extracts 11 features (identical to train_model.py),
and produces a list of timestamped packets for the simulation.
"""

import numpy as np
import pandas as pd

WINDOW_SIZE = 20  # readings per feature window (matches train_model.py)


def extract_features(window):
    """
    Extract 11-dimensional feature vector from a window DataFrame.
    Identical logic to train_model.py:10-38.
    """
    ax = window["accelerometerX"]
    ay = window["accelerometerY"]
    az = window["accelerometerZ"]
    speed = window["speed"]
    magnitude = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    return [
        ax.mean(),        ax.std(),         # 0: ax_mean,  1: ax_std
        ay.mean(),        ay.std(),         # 2: ay_mean,  3: ay_std
        az.mean(),        az.std(),         # 4: az_mean,  5: az_std
        magnitude.mean(), magnitude.std(),  # 6: mag_mean, 7: mag_std
        magnitude.max(),                    # 8: mag_max
        speed.mean(),     speed.std()       # 9: spd_mean, 10: spd_std
    ]


class SensorNode:
    """
    Represents one vehicle (one CSV file).
    node_id: globally unique 0-49 across all 50 vehicles.
    """

    def __init__(self, trip_id, vehicle_id, csv_path):
        self.trip_id = trip_id
        self.vehicle_id = vehicle_id
        self.node_id = (trip_id - 1) * 10 + (vehicle_id - 1)  # 0-49

        df = pd.read_csv(csv_path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Fill any NaN sensor values with 0 to avoid feature extraction errors
        for col in ["accelerometerX", "accelerometerY", "accelerometerZ",
                    "gyroX", "gyroY", "gyroZ", "speed"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        self.data = df

    def generate_packets(self):
        """
        Slide a non-overlapping window of WINDOW_SIZE over the data.
        Returns a list of packet dicts:
          {timestamp, lat, lon, features, sensor_id, trip_id, vehicle_id}
        """
        packets = []
        df = self.data
        n = len(df)
        for i in range(0, n - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = df.iloc[i: i + WINDOW_SIZE]
            features = extract_features(window)
            packets.append({
                "timestamp":  float(window["timestamp"].mean()),
                "lat":        float(window["latitude"].mean()),
                "lon":        float(window["longitude"].mean()),
                "features":   features,
                "sensor_id":  self.node_id,
                "trip_id":    self.trip_id,
                "vehicle_id": self.vehicle_id,
            })
        return packets
