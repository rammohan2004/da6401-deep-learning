"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_and_preprocess_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    '''parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('-l', '--loss', type=str, required=True,
                        choices=['mse', 'cross_entropy'])
    parser.add_argument('-o', '--optimizer', type=str, required=True,
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate', type=float, required=True)
    parser.add_argument('-wd', '--weight_decay', type=float, required=True)
    parser.add_argument('-nhl', '--num_layers', type=int, required=True)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', required=True)
    parser.add_argument('-a', '--activation', type=str, required=True,
                        choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, required=True,
                        choices=['random', 'xavier'])
    parser.add_argument('-w_p','--wandb_project', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='models/')'''

    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=50) # Use your best epoch count
    parser.add_argument('-b', '--batch_size', type=int, default=64) # Use your best batch size
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mse', 'cross_entropy'])
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'])
    parser.add_argument('-w_p','--wandb_project', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='models/')
    
    args = parser.parse_args()
    
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        raise ValueError(f"--hidden_size must have exactly {args.num_layers} values when specified as a list.")
    
    return args


def load_model(model_path):
    """
    Load trained model from disk.
    """
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'best_model.npy')
        
    weights_dict = np.load(model_path, allow_pickle=True).item()
    return weights_dict


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    y_pred = model.forward(X_test)
    y_pred = model.activations[-1].forward(y_pred)
    
    loss = model.loss_func.forward(y_test, y_pred)
    
    y_true_class = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    
    results = {
        'logits': y_pred,
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return results


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    print(f"Loading test data for {args.dataset}...")
    _, X_test, _, y_test = load_and_preprocess_data(args.dataset)
    
   # Instantiating the model directly with parsed args
    print("Initializing model...")
    model = NeuralNetwork(args)
    
    # Loading weights
    print(f"Loading weights from {args.model_path}...")
    weights_dict = load_model(args.model_path)
    
    # Putting weights into network using OOP method
    model.set_weights(weights_dict)
    
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    print("\n--- Test Set Results ---")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("------------------------")
    
    print("Evaluation complete!")
    return results

if __name__ == '__main__':
    main()