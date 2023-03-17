'''
    File name         : KalmanFilter.py
    Description       : KalmanFilter class used for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, u_z, std_acc, x_std_meas, y_std_meas, z_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """

        # Define sampling time
        self.dt = dt

        # Define the  control input variables
        self.u = np.matrix([u_x,u_y,u_z])

        # Intial State [x, dx, y, dy, z, dz]
        self.x = np.matrix([[0], 
                            [0], 
                            [0], 
                            [0], 
                            [0], 
                            [0]])

        # Define the State Transition Matrix A
        self.F = np.matrix([[1, self.dt, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, self.dt, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, self.dt],
                            [0, 0, 0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.G = np.matrix([[(self.dt**2)/2],
                            [self.dt],
                            [(self.dt**2)/2],
                            [self.dt],
                            [(self.dt**2)/2],
                            [self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])

        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, (self.dt**3)/2, 0, 0, 0, 0],
                            [(self.dt**3)/2, self.dt**2, 0, 0, 0, 0],
                            [0, 0, (self.dt**4)/4, (self.dt**3)/2, 0, 0],
                            [0, 0, (self.dt**3)/2, self.dt**2, 0, 0],
                            [0, 0, 0, 0, (self.dt**4)/4, (self.dt**3)/2],
                            [0, 0, 0, 0, (self.dt**3)/2, self.dt**2],]) * std_acc**2

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0, 0],
                           [0, y_std_meas**2, 0],
                           [0, 0, z_std_meas**2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.F.shape[1])

    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795

        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        print(np.shape(np.dot(self.F, self.x)), np.dot(self.F, self.x))
        print(np.shape(np.dot(self.G, self.u)), np.dot(self.G, self.u))
        # self.x = np.dot(self.F, self.x) + np.dot(self.G, self.u)
        # print(np.shape(self.x), self.x)
        self.x = np.dot(self.F, self.x) # disable control input

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return (self.x[0], self.x[2], self.x[4])

    def update(self, z):

        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)

        I = np.eye(self.H.shape[1])
        # temp = I - (K * self.H)
        temp = I - (K * self.H)

        # Update error covariance matrix
        self.P = temp * self.P * np.linalg.inv(temp) + K*self.R*K.T  #Eq.(13)
        return (self.x[0], self.x[2], self.x[4])
