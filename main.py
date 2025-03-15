import Kalman1D as kf
import yfinance as yf   
import matplotlib.pyplot as plt
import ParticleFilter as pf

if __name__ == '__main__':
    df = yf.download("0050.TW", period="1y", interval="1d")
    measurement = df["Close"].values[-250:]

    filtered = kf.Kalman(measurement)

    pf = pf.ParticleFilter(500, measurement[0], process_noise=1, measurement_noise=2)

    pf_filtered = []
    for price in measurement:
        pf.predict()
        pf.update(price)
        pf.resample()
        pf_filtered.append(pf.estimate())

    plt.plot(measurement, label="Original Prices", linewidth=2)
    plt.plot(filtered, label="Kalman Filtered Prices", linewidth=1, linestyle='--')
    plt.plot(pf_filtered, label="Particle Filtered Prices", linewidth=1, linestyle='dashdot')
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("0050 ETF Price Smoothing with Kalman Filter")
    plt.show()