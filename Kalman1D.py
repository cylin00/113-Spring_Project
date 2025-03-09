import numpy as np
from filterpy.kalman import KalmanFilter

def Kalman(measurement):

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

    return filtered