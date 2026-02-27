"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import NeuralLayer
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .objective_functions import CrossEntropy, MeanSquaredError
from .optimizers import SGD, Momentum, NAG, RMSprop, Adam, Nadam
import wandb


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
        elif cli_args.optimizer == 'adam':
            self.optimizer = Adam(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'nadam':
            self.optimizer = Nadam(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        
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
        out = X
        for i in range(len(self.layers)):
            z = self.layers[i].forward(out)
            a = self.activations[i].forward(z)
            out = a
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
        grad = self.loss_func.backward(y_true, y_pred)
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)
        
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update(self.layers)
    
    def train(self, X_train, y_train,X_val, y_val, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]
        #storing training loss, accuracy and validation loss, accuracy
        history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
        
        for epoch in range(epochs):

            #Shuffling the dataset
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, num_samples, batch_size):

                #SLicing the dataset to get a particular batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                #Forward pass
                y_pred = self.forward(X_batch)
                #Backward pass
                self.backward(y_batch, y_pred)
                #Updating weights
                self.update_weights()
            
            #Calculating epoch loss and accuracy
            epoch_loss, epoch_acc = self.evaluate(X_train, y_train)
            
            #Calculating validation loss, and logits
            val_logits =self.forward(X_val)
            val_loss =self.loss_func.forward(y_val, val_logits)
            
            #Calculating validation accuracy
            y_val_true_classes =np.argmax(y_val, axis=1)
            y_val_pred_classes = np.argmax(val_logits, axis=1)
            val_acc= np.mean(y_val_true_classes==y_val_pred_classes)
            
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
        
        #Calculating loss
        loss = self.loss_func.forward(y, y_pred)
        
        #converting to one hot encoding
        y_true_classes = np.argmax(y, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        #calculating accuaracy
        accuracy = np.mean(y_true_classes==y_pred_classes)
        
        return loss, accuracy
    
