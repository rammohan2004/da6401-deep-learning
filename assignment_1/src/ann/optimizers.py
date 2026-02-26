"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            
            #Adding weight decay
            layer.grad_W = layer.grad_W +self.weight_decay*layer.W

            #updating weights and biases
            layer.W= layer.W- self.learning_rate*layer.grad_W
            layer.b =layer.b -self.learning_rate*layer.grad_b

class Momentum:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, layers):
        for layer in layers:
            # Initial velocities are initialized to zero
            if layer not in self.velocities:
                self.velocities[layer] =[np.zeros_like(layer.W),np.zeros_like(layer.b)]
            
            # Adding weight deccay 
            layer.grad_W= layer.grad_W +(self.weight_decay *layer.W)
            
            # Getting old velocities
            v_w_old= self.velocities[layer][0]
            v_b_old =self.velocities[layer][1]
            
            #Calculating the new velocities
            v_w_new =self.momentum*v_w_old+ layer.grad_W
            v_b_new= self.momentum*v_b_old +layer.grad_b

            #Saving new velocities
            self.velocities[layer][0] =v_w_new
            self.velocities[layer][1]= v_b_new
            
            # Updating weights and biases
            layer.W= layer.W -self.learning_rate*v_w_new
            layer.b =layer.b- self.learning_rate*v_b_new

class NAG:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum =0.9
        self.weight_decay = weight_decay
        self.velocities = {} 

    def update(self, layers):
        for layer in layers:
            # Initial velocities are initialized to zero
            if layer not in self.velocities:
                self.velocities[layer] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            
            # Adding weight deccay 
            layer.grad_W= layer.grad_W +(self.weight_decay *layer.W)
            
            # Getting old velocities
            v_w_old = self.velocities[layer][0]
            v_b_old = self.velocities[layer][1]
            
           #Calculating the new velocities
            v_w_new = self.momentum*v_w_old +layer.grad_W
            v_b_new= self.momentum*v_b_old + layer.grad_b

            #Saving new velocities
            self.velocities[layer][0] = v_w_new
            self.velocities[layer][1]= v_b_new
            
             # Updating weights and biases
            layer.W =layer.W-self.learning_rate*(self.momentum*v_w_new+layer.grad_W)
            layer.b= layer.b-self.learning_rate*(self.momentum *v_b_new+ layer.grad_b)

class RMSprop:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta =0.9
        self.weight_decay = weight_decay
        self.S = {} 

    def update(self, layers):
        epsilon=1e-8
        for layer in layers:
            # Initial S list values are initialized to zero
            if layer not in self.S:
                self.S[layer] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            
            # Adding weight deccay 
            layer.grad_W = layer.grad_W +(self.weight_decay* layer.W)
            
            #Getting old values
            S_w_old = self.S[layer][0]
            S_b_old = self.S[layer][1]
            
            # Calculating new values
            S_w_new = self.beta*S_w_old +(1-self.beta)*(layer.grad_W**2)
            S_b_new = self.beta*S_b_old+ (1-self.beta)*(layer.grad_b**2)

            # Saving new S values
            self.S[layer][0] = S_w_new
            self.S[layer][1] = S_b_new

            #Updating old values
            layer.W = layer.W- (self.learning_rate/(np.sqrt(S_w_new)+epsilon))*layer.grad_W
            layer.b= layer.b -(self.learning_rate/(np.sqrt(S_b_new)+epsilon))*layer.grad_b


class Adam:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 =0.9
        self.beta2 =0.999
        self.weight_decay = weight_decay
        self.m = {} 
        self.v = {} 
        self.t = 0 

    def update(self, layers):
        epsilon=1e-8

        #incrementing step counter
        self.t += 1 
        
        for layer in layers:

            # Initialize initial values to zero
            if layer not in self.m:
                self.m[layer] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
                self.v[layer] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            
            # Adding weight decay
            layer.grad_W = layer.grad_W + (self.weight_decay * layer.W)

            #getting old values
            m_w= self.m[layer][0]
            m_b=self.m[layer][1]
            v_w= self.v[layer][0]
            v_b=self.v[layer][1]
            
            #Calculating new values
            m_w = self.beta1*m_w + (1-self.beta1)*layer.grad_W
            m_b=self.beta1*m_b + (1-self.beta1)*layer.grad_b
            v_w=self.beta2*v_w+(1-self.beta2)*(layer.grad_W**2)
            v_b=self.beta2*v_b+(1-self.beta2)*(layer.grad_b**2)

             # Saving new values
            self.m[layer][0]= m_w
            self.m[layer][1] = m_b
            self.v[layer][0]= v_w
            self.v[layer][1] =v_b
            
            # calculating bias corrected moments
            m_hat_w =m_w/(1-self.beta1**self.t)
            m_hat_b =m_b/(1-self.beta1**self.t)
            v_hat_w =v_w/(1-self.beta2**self.t)
            v_hat_b =v_b/(1-self.beta2**self.t)
            
            #Updating weights and biases
            layer.W =layer.W-(self.learning_rate/(np.sqrt(v_hat_w)+epsilon))*m_hat_w
            layer.b = layer.b-(self.learning_rate/(np.sqrt(v_hat_b)+epsilon))*m_hat_b


class Nadam:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 =0.9
        self.beta2 =0.999
        self.weight_decay = weight_decay
        self.m = {} 
        self.v = {} 
        self.t = 0  

    def update(self, layers):
        epsilon=1e-8

        #incrementing step counter
        self.t += 1 
        for layer in layers:
            # Initialize initial values to zero
            if layer not in self.m:
                self.m[layer] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
                self.v[layer] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            
            # Adding weight decay
            layer.grad_W = layer.grad_W + (self.weight_decay * layer.W)
            
            #getting old values
            m_w = self.m[layer][0]
            m_b=self.m[layer][1]
            v_w= self.v[layer][0]
            v_b= self.v[layer][1]
            
            #  #Calculating new values
            m_w = self.beta1*m_w+ (1 -self.beta1)*layer.grad_W
            m_b = self.beta1*m_b+ (1-self.beta1)* layer.grad_b
            v_w = self.beta2 * v_w + (1 - self.beta2) * (layer.grad_W ** 2)
            v_b = self.beta2 * v_b + (1 - self.beta2) * (layer.grad_b ** 2)

            # Saving new values
            self.m[layer][0] = m_w
            self.m[layer][1] =m_b
            self.v[layer][0] =v_w
            self.v[layer][1] = v_b
            
            #calculating bias corrected moments
            v_hat_w = v_w /(1 - self.beta2**self.t)
            v_hat_b = v_b / (1 - self.beta2**self.t)
            
            #Calculating nesterov bias-corrected first moment 
            m_hat_nesterov_w = (self.beta1 * m_w / (1 -self.beta1**self.t))+ ((1 -self.beta1)* layer.grad_W / (1 -self.beta1**self.t))
            m_hat_nesterov_b = (self.beta1 *m_b / (1-self.beta1**self.t)) + ((1 -self.beta1) * layer.grad_b /(1 - self.beta1**self.t))
            
            #Updating weights and biases
            layer.W = layer.W-(self.learning_rate/(np.sqrt(v_hat_w)+epsilon))*m_hat_nesterov_w
            layer.b = layer.b-(self.learning_rate/ (np.sqrt(v_hat_b) +epsilon))* m_hat_nesterov_b