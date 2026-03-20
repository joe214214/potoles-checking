"""
WSN Simulation Visualizer — ECE 659 Pothole Detection Project

Reads wsn_results.json produced by wsn_main.py and generates:
  1. wsn_map.html       — interactive Folium map on real OpenStreetMap tiles
  2. wsn_analysis.png   — matplotlib figure: queue analysis + network stats

Run from the project/ directory:
    python wsn_visualize.py
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import folium
from folium.plugins import HeatMap

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(_HERE, "wsn_results.json")
MAP_FILE     = os.path.join(_HERE, "wsn_map.html")
PLOT_FILE    = os.path.join(_HERE, "wsn_analysis.png")
DATA_DIR     = os.path.join(_HERE, "simulated_trips")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def load_vehicle_tracks():
    """Load GPS tracks for all 50 vehicles (downsampled for speed)."""
    tracks = []
    for trip_id in range(1, 6):
        for veh_id in range(1, 11):
            path = f"{DATA_DIR}/trip{trip_id}_vehicle{veh_id}.csv"
            if not os.path.exists(path):
                continue
            import pandas as pd
            df = pd.read_csv(path, usecols=["latitude", "longitude"])
            # Downsample: every 20th point
            df = df.iloc[::20].reset_index(drop=True)
            coords = list(zip(df["latitude"], df["longitude"]))
            tracks.append({"trip": trip_id, "vehicle": veh_id, "coords": coords})
    return tracks


# ─── 1. Folium map ────────────────────────────────────────────────────────────

TRIP_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]


def build_map(data):
    ch_positions = data["ch_positions"]         # [[lat, lon], ...]
    predictions  = data["predictions"]          # list of result dicts

    # Map centre = mean of all CH positions
    center_lat = float(np.mean([p[0] for p in ch_positions]))
    center_lon = float(np.mean([p[1] for p in ch_positions]))

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=13,
                   tiles="OpenStreetMap")

    # ── Vehicle trajectories (light gray) ────────────────────────────────────
    print("  Adding vehicle trajectories …")
    tracks = load_vehicle_tracks()
    for t in tracks:
        color = TRIP_COLORS[(t["trip"] - 1) % len(TRIP_COLORS)]
        folium.PolyLine(
            locations=t["coords"],
            color=color,
            weight=1.5,
            opacity=0.35,
            tooltip=f"Trip {t['trip']} Vehicle {t['vehicle']}",
        ).add_to(m)

    # ── Cluster Head markers ──────────────────────────────────────────────────
    for i, (lat, lon) in enumerate(ch_positions):
        folium.CircleMarker(
            location=[lat, lon],
            radius=12,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.9,
            tooltip=f"Cluster Head {i} (Trip {i+1})",
            popup=folium.Popup(
                f"<b>Cluster Head {i}</b><br>"
                f"Trip {i+1} centroid<br>"
                f"lat={lat:.5f}, lon={lon:.5f}",
                max_width=200,
            ),
        ).add_to(m)
        # CH label
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:white;'
                     f'text-align:center;line-height:24px;">CH{i}</div>',
                icon_size=(24, 24),
                icon_anchor=(12, 12),
            ),
        ).add_to(m)

    # ── Pothole predictions ───────────────────────────────────────────────────
    pothole_count = 0
    for r in predictions:
        if r["prediction"] != 1:
            continue
        pothole_count += 1
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.75,
            tooltip=(f"Pothole detected<br>"
                     f"CH{r['cluster_id']} | sensors={r['num_sensors']}"),
            popup=folium.Popup(
                f"<b>Pothole Detected</b><br>"
                f"Cluster: CH{r['cluster_id']}<br>"
                f"Sensors fused: {r['num_sensors']}<br>"
                f"lat={r['lat']:.5f}, lon={r['lon']:.5f}",
                max_width=220,
            ),
        ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;line-height:1.8">
      <b>ECE 659 WSN Pothole Detection</b><br>
      <span style="color:#1f77b4">&#9679;</span> Cluster Head (×5)<br>
      <span style="color:#d62728">&#9679;</span> Pothole detected (×{potholes})<br>
      <span style="color:#888">&#9135;</span> Vehicle trajectories (×50)
    </div>
    """.format(potholes=pothole_count)
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(MAP_FILE)
    print(f"  Map saved -> {MAP_FILE}  ({pothole_count} potholes shown)")
    return m


# ─── 2. matplotlib analysis figure ───────────────────────────────────────────

def build_plots(data):
    queue_logs  = data.get("ch_queue_logs", [])
    queue_stats = data.get("ch_queue_stats", [])
    pdr_list    = data.get("pdr_per_ch", [])
    net         = data.get("network_stats", {})

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ECE 659 WSN Simulation — Network Analysis", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (0,0) Queue length vs time ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("M/D/1 Queue Length over Time", fontsize=11)
    for log_entry in queue_logs:
        cid = log_entry["cluster_id"]
        log = log_entry["log"]
        if not log:
            continue
        ts  = [e["arrival_time"] for e in log]
        ql  = [e["queue_len"]    for e in log]
        # Shift time to start from 0
        t0 = ts[0]
        ax0.plot([t - t0 for t in ts], ql,
                 label=f"CH{cid}", linewidth=0.8, alpha=0.8)
    ax0.set_xlabel("Elapsed time (s)")
    ax0.set_ylabel("Queue length (packets)")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # ── (0,1) Waiting time histogram ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("M/D/1 Waiting Time Distribution", fontsize=11)
    for log_entry in queue_logs:
        cid = log_entry["cluster_id"]
        log = log_entry["log"]
        if not log:
            continue
        wt = [e["wait_time"] for e in log]
        ax1.hist(wt, bins=30, alpha=0.6, label=f"CH{cid}", density=True)
    ax1.set_xlabel("Waiting time (s)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── (0,2) Little's Law verification ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("Little's Law: $L_q$ vs $\\lambda \\cdot W_q$", fontsize=11)
    if queue_stats:
        labels = [f"CH{q['cluster_id']}" for q in queue_stats]
        Lq_sim = [q["Lq_sim"]           for q in queue_stats]
        LW_sim = [q["littles_L_sim"]     for q in queue_stats]
        x = np.arange(len(labels))
        w = 0.35
        ax2.bar(x - w/2, Lq_sim, w, label="$L_q$ (simulated)",  color="#4e79a7")
        ax2.bar(x + w/2, LW_sim, w, label="$\\lambda W_q$ (sim)", color="#f28e2b")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Avg queue length")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_title("Little's Law Verification", fontsize=11)

    # ── (1,0) M/D/1 utilisation + Lq comparison ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("M/D/1 $L_q$: Simulated vs Theoretical", fontsize=11)
    if queue_stats:
        labels     = [f"CH{q['cluster_id']}" for q in queue_stats]
        Lq_sim     = [q["Lq_sim"]     for q in queue_stats]
        Lq_theory  = [q["Lq_theory"]  for q in queue_stats]
        x = np.arange(len(labels))
        w = 0.35
        ax3.bar(x - w/2, Lq_sim,    w, label="Simulated",   color="#59a14f")
        ax3.bar(x + w/2, Lq_theory, w, label="Theoretical", color="#e15759",
                alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("Avg queue length $L_q$")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

    # ── (1,1) PDR per Cluster Head ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Packet Delivery Ratio per Cluster Head", fontsize=11)
    if pdr_list:
        labels = [f"CH{p['cluster_id']}" for p in pdr_list]
        pdrs   = [p["pdr"] * 100         for p in pdr_list]
        colors = [TRIP_COLORS[p["cluster_id"] % len(TRIP_COLORS)] for p in pdr_list]
        bars = ax4.bar(labels, pdrs, color=colors, edgecolor="white")
        ax4.set_ylim(0, 105)
        ax4.set_ylabel("PDR (%)")
        ax4.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, pdrs):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # ── (1,2) Overall network stats text box ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    total_gen = net.get("total_generated", 0)
    total_suc = net.get("total_channel_success", 0)
    overall_pdr = net.get("overall_pdr", 0)
    n_pred = len(data.get("predictions", []))
    n_pot  = sum(1 for r in data.get("predictions", []) if r["prediction"] == 1)

    text = (
        f"Network Summary\n"
        f"{'-' * 28}\n"
        f"Sensor nodes:       50\n"
        f"Cluster Heads:       5\n"
        f"Base Station:        1\n\n"
        f"Packets generated: {total_gen:>5}\n"
        f"Packets delivered: {total_suc:>5}\n"
        f"Overall PDR:      {overall_pdr:>5.1%}\n\n"
        f"BS inferences:    {n_pred:>5}\n"
        f"Potholes found:   {n_pot:>5}\n\n"
        f"Channel model: 900 MHz\n"
        f"Path-loss exp: n=2.5\n"
        f"MAC:           TDMA\n"
        f"Queue model:   M/D/1\n"
        f"Fusion:        Weighted avg\n"
        f"Classifier:    Random Forest"
    )
    ax5.text(0.05, 0.95, text, transform=ax5.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                       edgecolor="#cccccc"))

    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Analysis plot saved -> {PLOT_FILE}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. Run wsn_main.py first.")
        return

    print(f"Loading {RESULTS_FILE} …")
    data = load_results()

    print("Building Folium map …")
    build_map(data)

    print("Building analysis plots …")
    build_plots(data)

    print("\nDone!")
    print(f"  Open {MAP_FILE} in a browser to explore the interactive map.")
    print(f"  Open {PLOT_FILE} to view queue / network statistics.")


if __name__ == "__main__":
    main()
