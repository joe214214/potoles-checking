"""
WSN Simulation Main Runner — ECE 659 Pothole Detection Project

Realistic channel model:
  - Log-normal shadow fading  σ = 12 dB
  - Gilbert-Elliott two-state Markov per sensor (bursty loss)
  - CH queue overflow drop (buffer = 40 packets)

Outputs:
  wsn_results.json          — statistics (for wsn_visualize.py)
  wsn_animation_data.json   — timestamped events (for wsn_animate.py)

Run from anywhere:
    python wsn_main.py
"""

import glob
import json
import os
import numpy as np

from wsn.channel import WirelessChannel
from wsn.topology import Topology, haversine
from wsn.sensor_node import SensorNode
from wsn.cluster_head import ClusterHead
from wsn.base_station import BaseStation

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "simulated_trips")
MODEL_PATH = os.path.join(_HERE, "model", "pothole_rf_model.pkl")
RESULTS    = os.path.join(_HERE, "wsn_results.json")
ANIM_DATA  = os.path.join(_HERE, "wsn_animation_data.json")


def main():
    # ── 1. Load sensor data ───────────────────────────────────────────────────
    print("Loading sensor data ...")
    sensors = []
    for trip_id in range(1, 6):
        for veh_id in range(1, 11):
            path = os.path.join(DATA_DIR, f"trip{trip_id}_vehicle{veh_id}.csv")
            if os.path.exists(path):
                sensors.append(SensorNode(trip_id, veh_id, path))
    print(f"  {len(sensors)} sensor nodes loaded")

    # ── 2. Build network ──────────────────────────────────────────────────────
    print("Building topology ...")
    topology = Topology(DATA_DIR)
    cluster_heads = [
        ClusterHead(i, lat, lon)
        for i, (lat, lon) in enumerate(topology.ch_positions)
    ]
    base_station = BaseStation(MODEL_PATH)
    channel = WirelessChannel()

    print("  Cluster Head positions:")
    for i, (lat, lon) in enumerate(topology.ch_positions):
        print(f"    CH{i}: lat={lat:.5f}, lon={lon:.5f}")

    # ── 3. Generate packets ───────────────────────────────────────────────────
    print("Generating feature packets ...")
    all_packets = []
    for sensor in sensors:
        all_packets.extend(sensor.generate_packets())
    all_packets.sort(key=lambda p: p["timestamp"])
    print(f"  {len(all_packets)} packets generated")

    # ── 4. Simulation ─────────────────────────────────────────────────────────
    print("Simulating WSN transmission ...")

    stats = {
        "total_generated":        len(all_packets),
        "total_channel_attempts": 0,
        "total_channel_success":  0,
        "total_overflow_drops":   0,
        "ch_received":    [0] * 5,
        "ch_overflow":    [0] * 5,
        "ch_distances":   [[] for _ in range(5)],
    }

    # Animation event buffers
    tx_events     = []   # [t, from_lat, from_lon, to_lat, to_lon, ch_id, success(0/1)]
    queue_events  = []   # [t, ch_id, queue_len]

    for pkt in all_packets:
        # Nearest cluster head
        ch_id, dist_m = topology.get_nearest_ch(pkt["lat"], pkt["lon"])
        ch = cluster_heads[ch_id]

        # TDMA: compute actual transmission time
        t_tx = ch.tdma_tx_time(pkt["sensor_id"], pkt["timestamp"])
        pkt["t_tx"] = t_tx

        # Wireless channel (shadow fading + Markov)
        success, snr_db, pl_db = channel.transmit(dist_m,
                                                   sensor_id=pkt["sensor_id"])
        stats["total_channel_attempts"] += 1

        # Record TX event for animation (limit file size: keep all attempts)
        tx_events.append([
            round(t_tx, 2),
            round(pkt["lat"], 6), round(pkt["lon"], 6),
            round(ch.lat, 6),     round(ch.lon, 6),
            ch_id,
            1 if success else 0,
        ])

        if not success:
            continue

        # Packet reaches CH
        stats["total_channel_success"] += 1
        stats["ch_distances"][ch_id].append(dist_m)
        pkt["pl_db"]  = pl_db
        pkt["snr_db"] = snr_db

        accepted = ch.receive_packet(pkt, t_arrive=t_tx)
        if not accepted:
            # Queue overflow
            stats["total_overflow_drops"] += 1
            stats["ch_overflow"][ch_id]   += 1
            continue

        stats["ch_received"][ch_id] += 1

        # Queue snapshot for animation
        if ch.queue_log:
            last_q = ch.queue_log[-1]
            queue_events.append([round(t_tx, 2), ch_id, last_q["queue_len"]])

    # ── 5. Data fusion + ML inference ────────────────────────────────────────
    print("Running data fusion and pothole detection ...")
    pothole_events = []   # [t, lat, lon, ch_id]
    for ch in cluster_heads:
        for batch in ch.get_fusion_batches():
            result = base_station.receive_and_predict(batch)
            if result["prediction"] == 1:
                pothole_events.append([
                    round(result["timestamp"], 2),
                    round(result["lat"], 6),
                    round(result["lon"], 6),
                    result["cluster_id"],
                ])

    # ── 6. Summary ────────────────────────────────────────────────────────────
    pdr = stats["total_channel_success"] / max(stats["total_channel_attempts"], 1)
    n_pred     = len(base_station.predictions)
    n_potholes = len(pothole_events)

    print("\n" + "=" * 64)
    print("SIMULATION SUMMARY")
    print("=" * 64)
    print(f"  Packets generated          : {stats['total_generated']}")
    print(f"  Channel attempts           : {stats['total_channel_attempts']}")
    print(f"  Channel successes          : {stats['total_channel_success']}")
    print(f"  Overflow drops (CH buffer) : {stats['total_overflow_drops']}")
    print(f"  Overall PDR                : {pdr:.1%}")
    print(f"  BS inference calls         : {n_pred}")
    print(f"  Potholes detected          : {n_potholes}")

    print("\n  M/D/1 Queue Analysis (simulated vs theoretical):")
    print(f"  {'CH':>2} {'lam(pkt/s)':>10} {'rho':>6} "
          f"{'Lq_sim':>8} {'Lq_thy':>8} "
          f"{'Wq_sim(s)':>10} {'Wq_thy(s)':>10} "
          f"{'L=lam*W':>10} {'overflow':>10}")
    print("  " + "-" * 82)

    ch_stats_list = []
    for ch in cluster_heads:
        q = ch.queue_statistics()
        if q is None:
            print(f"  CH{ch.cluster_id}: no data")
            continue
        print(f"  CH{q['cluster_id']:>1}  "
              f"{q['lambda']:>10.4f} "
              f"{q['rho']:>6.3f} "
              f"{q['Lq_sim']:>8.4f} "
              f"{q['Lq_theory']:>8.4f} "
              f"{q['Wq_sim']:>10.4f} "
              f"{q['Wq_theory']:>10.4f} "
              f"{q['littles_L_sim']:>10.4f} "
              f"{ch.drops_overflow:>10d}")
        q["queue_log_sample"] = ch.queue_log
        ch_stats_list.append(q)

    print("\n  PDR per Cluster Head:")
    pdr_per_ch = []
    for i in range(5):
        arriving = sum(1 for p in all_packets
                       if topology.get_nearest_ch(p["lat"], p["lon"])[0] == i)
        pdr_i    = stats["ch_received"][i] / max(arriving, 1)
        avg_dist = (float(np.mean(stats["ch_distances"][i]))
                    if stats["ch_distances"][i] else 0.0)
        print(f"    CH{i}: received={stats['ch_received'][i]:4d}  "
              f"overflow={stats['ch_overflow'][i]:3d}  "
              f"PDR={pdr_i:.1%}  avg_dist={avg_dist:.0f} m")
        pdr_per_ch.append({
            "cluster_id":  i,
            "pdr":         pdr_i,
            "avg_dist_m":  avg_dist,
            "ch_received": stats["ch_received"][i],
            "overflow":    stats["ch_overflow"][i],
        })

    # ── 7. Save wsn_results.json ──────────────────────────────────────────────
    output = {
        "ch_positions":   topology.ch_positions,
        "predictions":    base_station.predictions,
        "network_stats":  {
            "total_generated":        stats["total_generated"],
            "total_channel_attempts": stats["total_channel_attempts"],
            "total_channel_success":  stats["total_channel_success"],
            "total_overflow_drops":   stats["total_overflow_drops"],
            "overall_pdr":            pdr,
            "ch_received":            stats["ch_received"],
        },
        "pdr_per_ch":     pdr_per_ch,
        "ch_queue_stats": [
            {k: v for k, v in q.items() if k != "queue_log_sample"}
            for q in ch_stats_list
        ],
        "ch_queue_logs": [
            {"cluster_id": q["cluster_id"],
             "log": q["queue_log_sample"][:2000]}
            for q in ch_stats_list
        ],
    }
    with open(RESULTS, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nStatistics saved  -> {RESULTS}")

    # ── 8. Save wsn_animation_data.json ──────────────────────────────────────
    all_tx_times = [e[0] for e in tx_events]
    t_start = float(min(all_tx_times)) if all_tx_times else 0.0
    t_end   = float(max(all_tx_times)) if all_tx_times else 1.0

    anim_out = {
        "t_start":       t_start,
        "t_end":         t_end,
        "ch_positions":  [[lat, lon] for lat, lon in topology.ch_positions],
        "tx_events":     [
            {"t": e[0], "from_lat": e[1], "from_lon": e[2],
             "to_lat": e[3], "to_lon": e[4],
             "ch_id": e[5], "success": bool(e[6])}
            for e in tx_events
        ],
        "queue_events":  [
            {"t": e[0], "ch_id": e[1], "queue_len": e[2]}
            for e in queue_events
        ],
        "pothole_events": [
            {"t": e[0], "lat": e[1], "lon": e[2], "ch_id": e[3]}
            for e in pothole_events
        ],
    }
    with open(ANIM_DATA, "w") as f:
        json.dump(anim_out, f, separators=(",", ":"))
    print(f"Animation data saved -> {ANIM_DATA}")
    print("\nRun  python wsn_animate.py   to generate the real-time animation.")
    print("Run  python wsn_visualize.py to generate the static analysis plots.")


if __name__ == "__main__":
    main()
