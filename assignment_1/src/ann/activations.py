"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


class ReLU:
    def forward(self, x):
        self.a = np.maximum(0, x)
        return self.a
    def backward(self, grad_output):
        derivative = (self.a > 0)
        local_error =  grad_output*derivative
        return local_error

class Sigmoid:
    
    def forward(self, x):
        self.a = 1 / (1 + np.exp(-x))
        return self.a
    def backward(self, grad_output):
        derivative = self.a*(1-self.a)
        local_error = grad_output*derivative
        return local_error
    
class Tanh:
    def forward(self, x):
        self.a = np.tanh(x)
        return self.a
    def backward(self, grad_output):
        derivative = 1-(self.a)**2
        local_error = grad_output*derivative
        return local_error
    
class Softmax:
    def forward(self, x):
        x_normal = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_normal)
        self.a = exps / np.sum(exps, axis=1, keepdims=True)
        return self.a
    def backward(self, grad_output):
        sum_grad_out = np.sum(grad_output * self.a, axis=1, keepdims=True)
        return self.a* (grad_output - sum_grad_out)

