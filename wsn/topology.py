"""
Network topology: LEACH-inspired geographic clustering.

5 Cluster Heads, one per trip, positioned at the GPS centroid
of all vehicles in that trip. Sensors dynamically associate
with the nearest CH using the Haversine distance formula.
"""

import math
import glob
import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two GPS coordinates."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


class Topology:
    """
    Computes 5 fixed Cluster Head positions (one per trip)
    and provides nearest-CH lookup for sensor nodes.
    """

    def __init__(self, data_dir):
        """
        data_dir: path to simulated_trips/ directory.
        CH position for trip i = mean(lat, lon) of all vehicles in trip i.
        """
        self.ch_positions = self._compute_ch_positions(data_dir)
        self.num_clusters = len(self.ch_positions)

    def _compute_ch_positions(self, data_dir):
        positions = []
        for trip_id in range(1, 6):
            files = glob.glob(f"{data_dir}/trip{trip_id}_vehicle*.csv")
            lats, lons = [], []
            for f in sorted(files):
                df = pd.read_csv(f, usecols=["latitude", "longitude"])
                lats.extend(df["latitude"].tolist())
                lons.extend(df["longitude"].tolist())
            if lats:
                positions.append((float(np.mean(lats)), float(np.mean(lons))))
        return positions  # [(lat0, lon0), ..., (lat4, lon4)]

    def get_nearest_ch(self, lat, lon):
        """
        Returns (cluster_id, distance_m) for the closest Cluster Head.
        """
        dists = [haversine(lat, lon, ch_lat, ch_lon)
                 for ch_lat, ch_lon in self.ch_positions]
        best = int(np.argmin(dists))
        return best, dists[best]
