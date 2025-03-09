import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# download 0050 ETF price, total # = 250
df = yf.download("0050.TW", period="1y", interval="1d")
measurement = df["Close"].values[-250:]

# price
kf = KalmanFilter(dim_x=1, dim_z=1)  

# random walk
kf.F = np.array([[1]])  

kf.H = np.array([[1]])  

# noise
kf.Q = np.array([[0.01]])  
kf.R = np.array([[1]])  

kf.x = np.array([[measurement[0]]])  
kf.P = np.array([[1]])  

filtered = []
for z in measurement:
    kf.predict()
    kf.update(z)
    filtered.append(kf.x[0, 0])

plt.plot(measurement, label="Original Prices", linewidth=2)
plt.plot(filtered, label="Kalman Filtered Prices", linewidth=1, linestyle='--')
plt.legend()
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("0050 ETF Price Smoothing with Kalman Filter")
plt.show()