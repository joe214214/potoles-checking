"""
Wireless channel model — ECE 659 WSN Simulation

Physical model (applied in order):
  1. Log-distance path loss  PL(d) = PL(d0) + 10·n·log10(d/d0)
  2. Log-normal shadow fading  X ~ N(0, σ²),  σ = SHADOW_STD_DB
  3. Gilbert-Elliott two-state Markov chain per sensor
       GOOD state: use path-loss + fading model
       BAD  state: forced PER = 0.95  (signal blocked / interference burst)
  4. BPSK BER in AWGN: BER = 0.5·erfc(√SNR_linear)
  5. Packet error rate: PER = 1−(1−BER)^L

Parameters chosen to give realistic PDR ≈ 60-80% for IoT at 200-500 m:
  P_tx = 20 dBm  (typical for 802.11p / DSRC short-range)
  P_noise = -87 dBm  (raised noise floor, reflects real IoT sensitivity)
  Shadow σ = 12 dB   (dense urban / suburban, includes obstacles)
  n = 2.5            (suburban path-loss exponent)
"""

import math
import numpy as np
from scipy.special import erfc


class WirelessChannel:
    # ── Radio parameters ──────────────────────────────────────────────────────
    F_C = 900e6        # Carrier: 900 MHz
    C   = 3e8          # Speed of light
    N   = 2.5          # Path-loss exponent
    D0  = 1.0          # Reference distance (m)
    P_TX_DBM   = 20    # Transmit power: 20 dBm
    P_NOISE_DBM = -87  # Effective noise/sensitivity floor: -87 dBm
    PACKET_BITS = 352  # 11 features × 32 bits

    # ── Shadow fading ─────────────────────────────────────────────────────────
    SHADOW_STD_DB = 12.0   # Log-normal shadow fading std (dB)

    # ── Gilbert-Elliott Markov channel ────────────────────────────────────────
    P_GOOD_TO_BAD = 0.03   # P(good→bad) per packet  (~3% chance of entering bad burst)
    P_BAD_TO_GOOD = 0.25   # P(bad→good) per packet  (~4 consecutive losses on average)
    PER_BAD_STATE = 0.95   # PER when link is blocked

    def __init__(self):
        self._PL0_dB = 20 * math.log10(
            4 * math.pi * self.D0 * self.F_C / self.C
        )
        # Per-sensor Markov state: 0 = GOOD, 1 = BAD
        self._sensor_state = {}   # sensor_id (int) → 0 or 1

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _path_loss_dB(self, d):
        d = max(d, self.D0)
        return self._PL0_dB + 10 * self.N * math.log10(d / self.D0)

    def _per_from_snr(self, snr_db):
        """BPSK in AWGN: BER = 0.5·erfc(√SNR_lin), PER = 1-(1-BER)^L."""
        snr_lin = 10 ** (snr_db / 10)
        ber = 0.5 * erfc(math.sqrt(max(snr_lin, 0.0)))
        ber = min(ber, 0.5)
        return 1.0 - (1.0 - ber) ** self.PACKET_BITS

    def _step_markov(self, sensor_id):
        """Advance Markov state for this sensor. Returns new state (0=good, 1=bad)."""
        state = self._sensor_state.get(sensor_id, 0)
        if state == 0:   # GOOD
            if np.random.random() < self.P_GOOD_TO_BAD:
                state = 1
        else:            # BAD
            if np.random.random() < self.P_BAD_TO_GOOD:
                state = 0
        self._sensor_state[sensor_id] = state
        return state

    # ── Public API ────────────────────────────────────────────────────────────

    def transmit(self, d, sensor_id=None):
        """
        Simulate one packet transmission from distance d (meters).
        sensor_id: used to track per-sensor Markov state (optional).
        Returns (success: bool, snr_dB: float, pl_dB: float).
        """
        pl_db = self._path_loss_dB(d)

        # 1. Log-normal shadow fading (random each call)
        shadow_db = np.random.normal(0, self.SHADOW_STD_DB)
        pl_effective = pl_db + shadow_db

        P_rx = self.P_TX_DBM - pl_effective
        snr_db = P_rx - self.P_NOISE_DBM

        # 2. Gilbert-Elliott Markov state
        if sensor_id is not None:
            state = self._step_markov(sensor_id)
            if state == 1:   # BAD — link blocked / burst interference
                success = np.random.random() > self.PER_BAD_STATE
                return success, snr_db, pl_effective

        # 3. BPSK path-loss + fading model
        per = self._per_from_snr(snr_db)
        success = np.random.random() > per
        return success, snr_db, pl_effective

    def path_loss_dB(self, d):
        """Deterministic path loss (no fading) — used for display/logging."""
        return self._path_loss_dB(d)
