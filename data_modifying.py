import pandas as pd
import numpy as np
import os

# 输入数据目录
input_dir = "Pothole"
output_dir = "simulated_trips"

os.makedirs(output_dir, exist_ok=True)

# 原始trip
trip_files = [
    "trip1_sensors.csv",
    "trip2_sensors.csv",
    "trip3_sensors.csv",
    "trip4_sensors.csv",
    "trip5_sensors.csv"
]

# 时间起点
base_time = 1700000000  # 任意统一时间戳

for trip_id, file in enumerate(trip_files):

    df = pd.read_csv(os.path.join(input_dir, file))

    for v in range(10):

        new_df = df.copy()

        # ===== 1 加速度扰动 =====
        for col in ["accelerometerX","accelerometerY","accelerometerZ"]:
            new_df[col] += np.random.normal(0,0.03,len(new_df))

        # ===== 2 速度扰动 =====
        new_df["speed"] += np.random.normal(0,0.3,len(new_df))

        # ===== 3 GPS微调 =====
        new_df["latitude"] += np.random.normal(0,0.0003,len(new_df))
        new_df["longitude"] += np.random.normal(0,0.0003,len(new_df))

        # ===== 4 时间统一 =====
        time_offset = np.random.randint(-30,30)

        new_df["timestamp"] = (
            new_df["timestamp"]
            - new_df["timestamp"].min()
            + base_time
            + time_offset
        )

        # 输出文件
        new_name = f"trip{trip_id+1}_vehicle{v+1}.csv"

        new_df.to_csv(os.path.join(output_dir,new_name),index=False)

print("Simulation finished.")