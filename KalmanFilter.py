import numpy as np
from filterpy.kalman import KalmanFilter

kf = KalmanFilter(dim_x=2, dim_z=1)

kf.F = np.array([[1, 1],
                 [0, 1]])

kf.H = np.array([[1, 0]])

kf.Q = np.array([[0.001, 0],
                 [0, 0.001]])

kf.R = np.array([[0.1]])

kf.x = np.array([[0],
                 [1]])

kf.P = np.array([[1, 0],
                 [0, 1]])

measurement = [i + np.random.normal(0, 0.1) for i in range(50)]

filtered = []

for z in measurement:
    kf.predict()
    kf.update(z)
    filtered.append(kf.x[0, 0])

import matplotlib
matplotlib.use("TkAgg")
print(matplotlib.get_backend())
import matplotlib.pyplot as plt

print("Measurement:", measurement[:5])
print("Filtered:", filtered[:5])

plt.plot(range(len(measurement)), measurement, label='Measurements', linewidth=2)
plt.plot(range(len(filtered)), filtered, label='Kalman Filter', linewidth=1, linestyle='--')
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("Kalman Filter Position Estimation")

plt.show()
print("Figure created:", plt.gcf())

