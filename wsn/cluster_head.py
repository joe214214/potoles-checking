"""
Cluster Head node.

Responsibilities:
  1. TDMA scheduling  — allocate time slots to sensor nodes.
  2. M/D/1 queue      — simulate deterministic-service queuing.
  3. Pre-filter       — coarse pothole check (high recall, broad threshold).
  4. Data fusion      — weighted average of feature vectors from multiple sensors.
"""

import numpy as np
from collections import defaultdict

# ── TDMA parameters ───────────────────────────────────────────────────────────
SLOT_DURATION = 0.3   # seconds (≈ sensor sampling interval)
NUM_SLOTS = 10        # slots per TDMA frame; one per vehicle within a trip

# ── M/D/1 queue parameters ────────────────────────────────────────────────────
SERVICE_TIME  = 0.1   # deterministic service time D (seconds)
MAX_QUEUE_LEN = 40    # maximum backlog before overflow drop (realistic buffer limit)

# ── Data fusion parameters ────────────────────────────────────────────────────
FUSION_WINDOW = 3.0   # group packets within this time window (seconds)


class ClusterHead:
    def __init__(self, cluster_id, lat, lon):
        self.cluster_id = cluster_id
        self.lat = lat
        self.lon = lon

        # ── TDMA ──────────────────────────────────────────────────────────────
        # Map sensor_id → assigned slot (0 .. NUM_SLOTS-1)
        self._slot_map = {}
        self._next_slot = 0

        # ── M/D/1 queue simulation ────────────────────────────────────────────
        # server_free_at: earliest time the processing core is available
        self._server_free_at = 0.0

        # Processed packets stored as:
        #   {"service_end": float, "packet": dict, "queue_len": int, "wait_time": float}
        self.processed = []

        # Raw queue stats (one entry per arrived-and-served packet)
        self.queue_log = []  # list of {"arrival_time", "wait_time", "queue_len"}

        # Running count of packets that have arrived (used for queue-length estimate)
        self._arrival_count = 0

        # Queue overflow drops counter
        self.drops_overflow = 0

    # ── TDMA ──────────────────────────────────────────────────────────────────

    def assign_slot(self, sensor_id):
        """Register a sensor and return its slot number (0-based)."""
        if sensor_id not in self._slot_map:
            self._slot_map[sensor_id] = self._next_slot % NUM_SLOTS
            self._next_slot += 1
        return self._slot_map[sensor_id]

    def tdma_tx_time(self, sensor_id, t_ready):
        """
        Return the earliest TDMA transmission time >= t_ready for this sensor.
        Sensors wait for their assigned slot; the frame repeats every
        NUM_SLOTS * SLOT_DURATION seconds.
        """
        slot = self.assign_slot(sensor_id)
        frame_dur = NUM_SLOTS * SLOT_DURATION
        # Start of the frame that contains t_ready
        frame_start = int(t_ready / frame_dur) * frame_dur
        t_tx = frame_start + slot * SLOT_DURATION
        if t_tx < t_ready:
            t_tx += frame_dur   # push to next frame
        return t_tx

    # ── M/D/1 queue ───────────────────────────────────────────────────────────

    def receive_packet(self, packet, t_arrive):
        """
        Simulate one packet entering the M/D/1 queue at t_arrive.
        Returns False if the queue is full (overflow drop), True otherwise.

        Computes:
          - queue length at arrival (number of packets still in service/waiting)
          - waiting time before service starts
          - time when service finishes (= when packet is "ready")
        """
        # Queue length at arrival: how many more packets can be served before
        # the arriving packet starts service.
        wait_backlog = max(0.0, self._server_free_at - t_arrive)
        queue_len = int(wait_backlog / SERVICE_TIME)

        # ── Queue overflow: drop if buffer is full ────────────────────────────
        if queue_len >= MAX_QUEUE_LEN:
            self.drops_overflow += 1
            return False   # packet dropped

        service_start = max(t_arrive, self._server_free_at)
        service_end = service_start + SERVICE_TIME
        wait_time = service_start - t_arrive

        self._server_free_at = service_end
        self._arrival_count += 1

        self.queue_log.append({
            "arrival_time": t_arrive,
            "wait_time":    wait_time,
            "queue_len":    queue_len,
        })
        self.processed.append({
            "service_end": service_end,
            "packet":      packet,
            "wait_time":   wait_time,
            "queue_len":   queue_len,
        })
        return True   # packet accepted

    # ── Pre-filter ────────────────────────────────────────────────────────────

    @staticmethod
    def pre_filter(features):
        """
        Coarse pothole check at the cluster head.
        Thresholds are intentionally broad (high recall).
        Only data that clearly cannot be a pothole is discarded.

        features[8] = magnitude.max()   (acceleration spike)
        features[5] = az.std()          (vertical jitter)
        """
        mag_max = features[8]
        az_std  = features[5]
        return mag_max > 1.2 or az_std > 0.15

    # ── Data fusion ───────────────────────────────────────────────────────────

    @staticmethod
    def fuse_features(packets):
        """
        Weighted-average fusion.
        Weight for each packet = 1 / pl_db  (lower path loss → better signal
        → higher contribution).
        Returns fused 11-dim feature vector.
        """
        features_arr = np.array([p["features"] for p in packets], dtype=float)
        weights = np.array([1.0 / max(p["pl_db"], 1.0) for p in packets])
        weights /= weights.sum()
        return weights @ features_arr   # shape (11,)

    # ── Batch fusion (called after simulation) ────────────────────────────────

    def get_fusion_batches(self):
        """
        Group all processed-and-filtered packets into FUSION_WINDOW buckets.
        Returns a list of dicts ready for BaseStation.receive_and_predict().
        """
        if not self.processed:
            return []

        # Bucket by floor(service_end / FUSION_WINDOW)
        buckets = defaultdict(list)
        for item in self.processed:
            key = int(item["service_end"] / FUSION_WINDOW)
            buckets[key].append(item["packet"])

        batches = []
        for key in sorted(buckets):
            raw_packets = buckets[key]

            # Apply pre-filter
            filtered = [p for p in raw_packets if self.pre_filter(p["features"])]
            if not filtered:
                continue

            fused = self.fuse_features(filtered)
            avg_lat = float(np.mean([p["lat"] for p in filtered]))
            avg_lon = float(np.mean([p["lon"] for p in filtered]))
            avg_t   = float(np.mean([p["timestamp"] for p in filtered]))

            batches.append({
                "features":    fused.tolist(),
                "lat":         avg_lat,
                "lon":         avg_lon,
                "timestamp":   avg_t,
                "cluster_id":  self.cluster_id,
                "num_sensors": len(filtered),
            })
        return batches

    # ── M/D/1 theoretical statistics ─────────────────────────────────────────

    def queue_statistics(self):
        """
        Compute simulated vs theoretical M/D/1 queue metrics.
        Returns a dict with lambda, rho, Lq_sim, Lq_theory, Wq_sim, Wq_theory,
        and Little's Law check values.
        """
        if len(self.queue_log) < 2:
            return None

        times = [e["arrival_time"] for e in self.queue_log]
        T_obs = times[-1] - times[0]
        if T_obs <= 0:
            return None

        lam = len(self.queue_log) / T_obs      # arrival rate (packets/s)
        mu  = 1.0 / SERVICE_TIME               # service rate (packets/s)
        rho = lam / mu                         # utilisation

        wait_times  = [e["wait_time"]  for e in self.queue_log]
        queue_lens  = [e["queue_len"]  for e in self.queue_log]

        Lq_sim = float(np.mean(queue_lens))
        Wq_sim = float(np.mean(wait_times))

        if rho < 1.0:
            Lq_theory = rho ** 2 / (2 * (1 - rho))
            Wq_theory = rho / (2 * mu * (1 - rho))
        else:
            Lq_theory = Wq_theory = float("inf")

        return {
            "cluster_id":    self.cluster_id,
            "n_packets":     len(self.queue_log),
            "lambda":        lam,
            "rho":           rho,
            "Lq_sim":        Lq_sim,
            "Lq_theory":     Lq_theory,
            "Wq_sim":        Wq_sim,
            "Wq_theory":     Wq_theory,
            # Little's Law: L = λ·W
            "littles_L_sim":    lam * Wq_sim,
            "littles_L_theory": lam * Wq_theory if rho < 1 else float("inf"),
        }
