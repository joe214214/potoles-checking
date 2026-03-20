import os
import json
import math
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None


# =========================================================
# 1. 基本配置（你可以改）
# =========================================================
USE_SIMULATED = True                 # True: 读 simulated_trips/，False: 读 Pothole/
WINDOW_SIZE = 20                    # 一个窗口多少个 sample
STEP_SIZE = 10                      # 滑动步长
UPDATE_INTERVAL_SECONDS = 10        # 地图每隔多少秒“更新一次”
POTHOLE_THRESHOLD = 0.65            # 模型概率阈值
MIN_SPEED = 1.0                     # 太慢的车忽略，减少误报
MAX_EVENTS_PER_BATCH = 15           # 每个时间批次最多显示多少个点，避免太密
OUTPUT_HTML = "dynamic_pothole_map.html"

# 如果没有模型，用 heuristic 规则兜底
USE_HEURISTIC_FALLBACK = True


# =========================================================
# 2. 路径
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "simulated_trips" if USE_SIMULATED else "Pothole")
MODEL_PATH = os.path.join(BASE_DIR, "model", "pothole_rf_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_HTML)


# =========================================================
# 3. 特征提取
# =========================================================
def extract_features(window: pd.DataFrame):
    ax = window["accelerometerX"].to_numpy()
    ay = window["accelerometerY"].to_numpy()
    az = window["accelerometerZ"].to_numpy()
    speed = window["speed"].to_numpy()

    mag = np.sqrt(ax**2 + ay**2 + az**2)

    features = [
        ax.mean(),
        ax.std(ddof=0),
        ay.mean(),
        ay.std(ddof=0),
        az.mean(),
        az.std(ddof=0),
        mag.mean(),
        mag.std(ddof=0),
        mag.max(),
        speed.mean(),
        speed.std(ddof=0),
    ]
    return features


FEATURE_NAMES = [
    "ax_mean", "ax_std",
    "ay_mean", "ay_std",
    "az_mean", "az_std",
    "mag_mean", "mag_std", "mag_max",
    "speed_mean", "speed_std"
]


# =========================================================
# 4. 读取模型
# =========================================================
def load_model():
    if joblib is None:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    obj = joblib.load(MODEL_PATH)

    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


# =========================================================
# 5. heuristic 兜底
#    没有模型时，用简单规则模拟“服务器判断”
# =========================================================
def heuristic_pothole_score(window: pd.DataFrame):
    ax = window["accelerometerX"].to_numpy()
    ay = window["accelerometerY"].to_numpy()
    az = window["accelerometerZ"].to_numpy()
    speed = window["speed"].to_numpy()

    mag = np.sqrt(ax**2 + ay**2 + az**2)

    # 一些简单统计
    mag_std = float(np.std(mag))
    mag_max = float(np.max(mag))
    speed_mean = float(np.mean(speed))

    # 这里是经验型规则，不是严格物理模型
    # 目的只是：没有训练模型时也能跑起来
    score = 0.0

    # 波动越大越像坑洞
    score += min(mag_std / 0.20, 1.0) * 0.45

    # 峰值越高越像坑洞
    score += min((mag_max - 1.1) / 0.8, 1.0) * 0.40

    # 速度太低时，置信度稍降
    if speed_mean < MIN_SPEED:
        score *= 0.3
    else:
        score += min(speed_mean / 10.0, 1.0) * 0.15

    return max(0.0, min(score, 1.0))


# =========================================================
# 6. 枚举数据文件
# =========================================================
def get_sensor_files():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    # simulated_trips 里一般全是 vehicle csv
    # Pothole 里要排除 *_potholes.csv
    files = [f for f in files if not f.endswith("_potholes.csv")]

    if not files:
        raise FileNotFoundError(f"No sensor CSV files found in: {DATA_DIR}")

    return files


# =========================================================
# 7. 从每辆车提取“疑似坑洞事件”
# =========================================================
def detect_events_from_file(file_path, model=None):
    df = pd.read_csv(file_path)

    required_cols = {
        "timestamp", "latitude", "longitude", "speed",
        "accelerometerX", "accelerometerY", "accelerometerZ"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(file_path)} missing columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)

    events = []
    vehicle_id = os.path.splitext(os.path.basename(file_path))[0]

    for start_idx in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df.iloc[start_idx:start_idx + WINDOW_SIZE].copy()

        speed_mean = float(window["speed"].mean())
        if speed_mean < MIN_SPEED:
            continue

        feats = extract_features(window)

        if model is not None:
            x = np.array(feats, dtype=float).reshape(1, -1)

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(x)[0, 1])
            else:
                pred = int(model.predict(x)[0])
                prob = 1.0 if pred == 1 else 0.0
        else:
            if not USE_HEURISTIC_FALLBACK:
                continue
            prob = heuristic_pothole_score(window)

        if prob >= POTHOLE_THRESHOLD:
            mid = window.iloc[len(window) // 2]

            events.append({
                "vehicle_id": vehicle_id,
                "timestamp": float(mid["timestamp"]),
                "latitude": float(mid["latitude"]),
                "longitude": float(mid["longitude"]),
                "probability": round(prob, 4),
                "speed_mean": round(speed_mean, 3),
            })

    return events


# =========================================================
# 8. 组装所有车辆事件
# =========================================================
def collect_all_events():
    model = load_model()
    files = get_sensor_files()

    all_events = []
    trajectories = []

    for f in files:
        df = pd.read_csv(f).sort_values("timestamp").reset_index(drop=True)

        # 轨迹点稀疏采样，用于地图上画淡线
        traj_sample = df.iloc[::max(1, len(df) // 200 + 1)][["latitude", "longitude"]].copy()
        traj_points = traj_sample.values.tolist()

        trajectories.append({
            "vehicle_id": os.path.splitext(os.path.basename(f))[0],
            "points": traj_points
        })

        events = detect_events_from_file(f, model=model)
        all_events.extend(events)

    if not all_events:
        raise RuntimeError("No pothole events detected. Try lowering POTHOLE_THRESHOLD.")

    events_df = pd.DataFrame(all_events).sort_values("timestamp").reset_index(drop=True)

    # 把时间归一化到一个统一的模拟起点
    t0 = float(events_df["timestamp"].min())
    sim_start = datetime(2026, 1, 1, 8, 0, 0)

    def to_sim_time(ts):
        delta_sec = ts - t0
        return sim_start + timedelta(seconds=float(delta_sec))

    events_df["sim_time"] = events_df["timestamp"].apply(to_sim_time)

    return events_df, trajectories


# =========================================================
# 9. 按 update interval 分批
# =========================================================
def make_batches(events_df: pd.DataFrame):
    sim_start = events_df["sim_time"].min()

    delta_sec = (events_df["sim_time"] - sim_start).dt.total_seconds()
    batch_id = (delta_sec // UPDATE_INTERVAL_SECONDS).astype(int)

    events_df = events_df.copy()
    events_df["batch_id"] = batch_id

    batches = []

    for bid, grp in events_df.groupby("batch_id"):
        grp = grp.sort_values("probability", ascending=False).head(MAX_EVENTS_PER_BATCH)

        batch_time = sim_start + timedelta(seconds=int(bid * UPDATE_INTERVAL_SECONDS))

        items = []
        for _, row in grp.iterrows():
            items.append({
                "lat": row["latitude"],
                "lon": row["longitude"],
                "vehicle_id": row["vehicle_id"],
                "prob": float(row["probability"]),
                "speed_mean": float(row["speed_mean"]),
                "time_str": row["sim_time"].strftime("%H:%M:%S"),
            })

        batches.append({
            "batch_id": int(bid),
            "batch_time": batch_time.strftime("%H:%M:%S"),
            "items": items
        })

    return batches


# =========================================================
# 10. 生成 HTML 页面
# =========================================================
def build_html(center_lat, center_lon, batches, trajectories):
    batches_json = json.dumps(batches, ensure_ascii=False)
    traj_json = json.dumps(trajectories, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dynamic Pothole Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <link
      rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    />

    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
        }}

        #container {{
            display: flex;
            width: 100%;
            height: 100%;
        }}

        #map {{
            flex: 1;
            height: 100%;
        }}

        #panel {{
            width: 340px;
            height: 100%;
            border-left: 1px solid #ccc;
            box-sizing: border-box;
            padding: 14px;
            overflow-y: auto;
            background: #fafafa;
        }}

        .title {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .meta {{
            font-size: 14px;
            margin-bottom: 8px;
            color: #333;
        }}

        .controls {{
            margin: 12px 0;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        button {{
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            background: #1976d2;
            color: white;
        }}

        button:hover {{
            background: #145ca3;
        }}

        .log-item {{
            border: 1px solid #ddd;
            background: white;
            padding: 8px;
            margin-bottom: 8px;
            border-radius: 8px;
            font-size: 13px;
        }}

        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
            background: #ef5350;
            color: white;
            margin-left: 6px;
        }}

        .legend {{
            margin-top: 10px;
            font-size: 13px;
            color: #444;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
<div id="container">
    <div id="map"></div>

    <div id="panel">
        <div class="title">Dynamic Pothole Map</div>

        <div class="meta" id="currentTime">Current update time: --:--:--</div>
        <div class="meta" id="batchInfo">Current batch: 0</div>
        <div class="meta" id="totalPoints">Accumulated points: 0</div>

        <div class="controls">
            <button onclick="startPlayback()">Start</button>
            <button onclick="pausePlayback()">Pause</button>
            <button onclick="stepPlayback()">Step</button>
            <button onclick="resetPlayback()">Reset</button>
        </div>

        <div class="legend">
            <div><b>Display logic</b></div>
            <div>• Map updates every batch</div>
            <div>• Newly detected suspicious pothole points are appended</div>
            <div>• Old points remain on the map</div>
            <div>• Redder / larger point = higher pothole confidence</div>
        </div>

        <hr>

        <div id="log"></div>
    </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
    const batches = {batches_json};
    const trajectories = {traj_json};

    const map = L.map('map').setView([{center_lat}, {center_lon}], 14);

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    // 画淡轨迹，帮助用户看“城市范围”
    trajectories.forEach(tr => {{
        if (tr.points && tr.points.length > 1) {{
            L.polyline(tr.points, {{
                color: '#607d8b',
                weight: 1,
                opacity: 0.20
            }}).addTo(map);
        }}
    }});

    let currentBatchIndex = 0;
    let timer = null;
    let totalPoints = 0;

    function colorByProb(prob) {{
        if (prob >= 0.9) return '#b71c1c';
        if (prob >= 0.8) return '#d32f2f';
        if (prob >= 0.7) return '#f44336';
        return '#ff8a65';
    }}

    function radiusByProb(prob) {{
        return 5 + Math.round(prob * 8);
    }}

    function addBatch(batch) {{
        document.getElementById('currentTime').innerText =
            'Current update time: ' + batch.batch_time;

        document.getElementById('batchInfo').innerText =
            'Current batch: ' + batch.batch_id;

        batch.items.forEach(item => {{
            const marker = L.circleMarker([item.lat, item.lon], {{
                radius: radiusByProb(item.prob),
                color: colorByProb(item.prob),
                fillColor: colorByProb(item.prob),
                fillOpacity: 0.65,
                weight: 1
            }}).addTo(map);

            marker.bindPopup(
                '<b>Suspected pothole</b><br>' +
                'Vehicle: ' + item.vehicle_id + '<br>' +
                'Time: ' + item.time_str + '<br>' +
                'Probability: ' + item.prob.toFixed(3) + '<br>' +
                'Mean speed: ' + item.speed_mean.toFixed(2)
            );

            totalPoints += 1;
        }});

        document.getElementById('totalPoints').innerText =
            'Accumulated points: ' + totalPoints;

        const log = document.getElementById('log');
        const div = document.createElement('div');
        div.className = 'log-item';
        div.innerHTML =
            '<b>Update ' + batch.batch_id + '</b>' +
            '<span class="badge">+' + batch.items.length + '</span><br>' +
            'Server time: ' + batch.batch_time + '<br>' +
            'New suspicious points appended to map.';
        log.prepend(div);
    }}

    function playOneStep() {{
        if (currentBatchIndex >= batches.length) {{
            pausePlayback();
            return;
        }}

        addBatch(batches[currentBatchIndex]);
        currentBatchIndex += 1;
    }}

    function startPlayback() {{
        if (timer !== null) return;
        timer = setInterval(playOneStep, 1200);   // 每 1.2 秒播放一个批次（只是演示速度）
    }}

    function pausePlayback() {{
        if (timer !== null) {{
            clearInterval(timer);
            timer = null;
        }}
    }}

    function stepPlayback() {{
        pausePlayback();
        playOneStep();
    }}

    function resetPlayback() {{
        location.reload();
    }}
</script>
</body>
</html>
"""
    return html


# =========================================================
# 11. 主函数
# =========================================================
def main():
    events_df, trajectories = collect_all_events()
    batches = make_batches(events_df)

    center_lat = float(events_df["latitude"].mean())
    center_lon = float(events_df["longitude"].mean())

    html = build_html(center_lat, center_lon, batches, trajectories)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Done. HTML saved to: {OUTPUT_PATH}")
    print(f"Total detected events: {len(events_df)}")
    print(f"Total batches: {len(batches)}")
    print(f"Map center: ({center_lat:.6f}, {center_lon:.6f})")


if __name__ == "__main__":
    main()