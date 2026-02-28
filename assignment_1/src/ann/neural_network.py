"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import NeuralLayer
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .objective_functions import CrossEntropy, MeanSquaredError
from .optimizers import SGD, Momentum, NAG, RMSprop
import wandb
from sklearn.metrics import f1_score

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.args = cli_args

        #input image 28 X 28
        self.input_dim = 784

        #output total 10 classes
        self.output_dim = 10
        
        # Initializing loss
        if cli_args.loss == 'cross_entropy':
            self.loss_func = CrossEntropy()
        elif cli_args.loss == 'mse':
            self.loss_func=MeanSquaredError()
        
        # Initializing Optimizer
        if cli_args.optimizer == 'sgd':
            self.optimizer = SGD(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'momentum':
            self.optimizer = Momentum(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'nag':
            self.optimizer = NAG(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
    
        #Building network architecture
        self.layers = []
        self.activations = []
        
        #Mapping activations
        activation_map = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}
        ActivationClass = activation_map[cli_args.activation]
        
        current_input_dim = self.input_dim
        
        # Looping through list of hidden layer
        for hidden_nodes in cli_args.hidden_size:
            
            #Adding neural layer
            self.layers.append(NeuralLayer(cli_args.weight_init, current_input_dim, hidden_nodes))
            #Adding corresponding activation
            self.activations.append(ActivationClass())
            #Updating input dimension  for next layer
            current_input_dim =hidden_nodes
            
        #Adding output layer
        self.layers.append(NeuralLayer(cli_args.weight_init, current_input_dim, self.output_dim))
        #Adding softmax for output layer
        self.activations.append(Softmax())
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        out = X
        for i in range(len(self.layers)):
            z = self.layers[i].forward(out)
            if i != (len(self.layers)-1):
             z = self.activations[i].forward(z)
            out = z
        return out
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """

        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """

        grad_W_list = []
        grad_b_list = []

         # Backprop through layers in reverse; collect grads so that index 0 = last layer

        grad = self.loss_func.backward(y_true, y_pred)
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)

            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)
        
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[0].shape)
        print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[0].shape)
        return self.grad_W, self.grad_b

        
        
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update(self.layers)
    
    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]
        val_size = int(0.1 * num_samples) 
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        X_val = X_shuffled[:val_size]
        y_val = y_shuffled[:val_size]
        
        X_train_split = X_shuffled[val_size:]
        y_train_split = y_shuffled[val_size:]
        
        train_samples = X_train_split.shape[0] 
        #storing training loss, accuracy and validation loss, accuracy
        history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}

        best_val_f1=-1
        self.best_weights=None
        
        for epoch in range(epochs):

            #Shuffling the dataset
            indices = np.random.permutation(train_samples)
            X_epoch = X_train_split[indices]
            y_epoch = y_train_split[indices]
            
            for i in range(0, num_samples, batch_size):

                #SLicing the dataset to get a particular batch
                X_batch = X_epoch[i:i+batch_size]
                y_batch = y_epoch[i:i+batch_size]
                
                #Forward pass
                y_pred = self.forward(X_batch)
                #applying softmax activation
                y_pred = self.activations[-1].forward(y_pred)

                #Backward pass
                self.backward(y_batch, y_pred)
                #Updating weights
                self.update_weights()
            
            #Calculating epoch loss and accuracy
            epoch_loss, epoch_acc, _ = self.evaluate(X_train_split, y_train_split)
            
            #Calculating validation loss, and logits
            val_logits =self.forward(X_val)
            val_logits=self.activations[-1].forward(val_logits)
            val_loss =self.loss_func.forward(y_val, val_logits)
            
            #Calculating validation accuracy
            y_val_true_classes =np.argmax(y_val, axis=1)
            y_val_pred_classes = np.argmax(val_logits, axis=1)
            val_acc= np.mean(y_val_true_classes==y_val_pred_classes)
            val_f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='macro', zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_weights = self.get_weights()
            # Save to history dictionary
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            #wandb log
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_logits': wandb.Histogram(val_logits) 
                })
            
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
        return history
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        #Forward method for predictions
        y_pred = self.forward(X)
        #applying softmax activation
        y_pred = self.activations[-1].forward(y_pred)
        #Calculating loss
        loss = self.loss_func.forward(y, y_pred)
        
        #converting to one hot encoding
        y_true_classes = np.argmax(y, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        #calculating accuaracy
        accuracy = np.mean(y_true_classes==y_pred_classes)
        f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        
        return loss, accuracy, f1
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
