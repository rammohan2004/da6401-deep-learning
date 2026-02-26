"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class CrossEntropy:
    def forward(self, y_true, y_pred):
        epsilon = 1e-9
        N = y_true.shape[0]
        total_sum = np.sum(y_true*(np.log(y_pred+epsilon)))
        return -total_sum/N
    def backward(self, y_true, y_pred):
        N = y_true.shape[0]
        epsilon = 1e-9
        derivative = -(y_true/(y_pred+epsilon))/N
        return derivative


class MeanSquaredError:
    def forward(self, y_true, y_pred):
        N = y_true.shape[0]
        total_sum = np.sum((y_pred-y_true)**2)
        return total_sum/N
    def backward(self, y_true, y_pred):
        diff = y_pred - y_true
        N = y_true.shape[0]
        derivative =  (2/N)*(diff)
        return derivative