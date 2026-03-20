import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

WINDOW_SIZE = 20

def extract_features(window):

    ax = window["accelerometerX"]
    ay = window["accelerometerY"]
    az = window["accelerometerZ"]
    speed = window["speed"]

    magnitude = np.sqrt(ax**2 + ay**2 + az**2)

    features = [

        ax.mean(),
        ax.std(),

        ay.mean(),
        ay.std(),

        az.mean(),
        az.std(),

        magnitude.mean(),
        magnitude.std(),
        magnitude.max(),

        speed.mean(),
        speed.std()
    ]

    return features


X = []
y = []

sensor_files = glob.glob("Pothole/*_sensors.csv")

for file in sensor_files:

    pothole_file = file.replace("_sensors","_potholes")

    sensor = pd.read_csv(file)
    potholes = pd.read_csv(pothole_file)

    pothole_times = potholes["timestamp"].values

    for i in range(0, len(sensor)-WINDOW_SIZE):

        window = sensor.iloc[i:i+WINDOW_SIZE]

        feature = extract_features(window)

        start = window["timestamp"].min()
        end = window["timestamp"].max()

        label = 0

        for t in pothole_times:
            if start <= t <= end:
                label = 1
                break

        X.append(feature)
        y.append(label)


X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print(classification_report(y_test,pred))
import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/pothole_rf_model.pkl")

print("Model saved to model/pothole_rf_model.pkl")