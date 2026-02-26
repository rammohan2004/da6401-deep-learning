"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class NeuralLayer:
    def __init__(self,weight_init_method, input_nodes, output_nodes):

        if weight_init_method == "random":
            self.W = np.random.randn(input_nodes, output_nodes)*0.01
        elif weight_init_method == "xavier":
            scaler = (2/(input_nodes+output_nodes))**(0.5)
            self.W = np.random.randn(input_nodes, output_nodes)*scaler
        self.b = np.zeros((1, output_nodes))
        self.grad_W = None
        self.grad_b = None
        self.X = None

    def forward(self, X):
        self.X = X
        Z = X @ self.W + self.b
        return Z
    def backward(self, grad_output):
        self.grad_W = (self.X).T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_input =  grad_output @ (self.W).T
        return grad_input