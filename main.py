import Kalman1D as kf
import yfinance as yf   
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = yf.download("0050.TW", period="1y", interval="1d")
    measurement = df["Close"].values[-250:]

    filtered = kf.Kalman(measurement)

    plt.plot(measurement, label="Original Prices", linewidth=2)
    plt.plot(filtered, label="Kalman Filtered Prices", linewidth=1, linestyle='--')
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("0050 ETF Price Smoothing with Kalman Filter")
    plt.show()