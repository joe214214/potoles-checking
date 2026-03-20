import joblib
import numpy as np

# load model
data = joblib.load("pothole_rf_model.pkl")

model = data["model"]

features = np.array([[
    ax_mean,
    ax_std,
    ay_mean,
    ay_std,
    az_mean,
    az_std,
    mag_mean,
    mag_std,
    mag_max,
    speed_mean,
    speed_std
]])

prediction = model.predict(features)

print(prediction)