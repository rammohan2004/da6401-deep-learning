"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.data_loader import load_and_preprocess_data
from src.ann.neural_network import NeuralNetwork

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
    parser = argparse.ArgumentParser(description='Run inference on test set')

    
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model directory or .npy file')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_layers', type=int, required=True, help='Number of hidden layers')
    parser.add_argument('--num_neurons', type=int, nargs='+', required=True, help='List of hidden layer sizes')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], required=True)
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'model.npy')
        
    weights_dict = np.load(model_path, allow_pickle=True).item()
    return weights_dict


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    y_pred = model.forward(X_test)
    
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
    _, _, X_test, _, _, y_test = load_and_preprocess_data(args.dataset)
    
    #Reconstructing 
    class DummyArgs:
        def __init__(self, inference_args):
            self.loss = 'cross_entropy'
            self.optimizer = 'sgd'
            self.learning_rate = 0.01
            self.weight_decay = 0.0
            self.activation = inference_args.activation
            self.num_layers = inference_args.hidden_layers
            self.hidden_size = inference_args.num_neurons
            self.weight_init = 'random'
            
    dummy_args = DummyArgs(args)
    

    if len(dummy_args.hidden_size) == 1:
        dummy_args.hidden_size = dummy_args.hidden_size * dummy_args.num_layers
        
    # Instanitiating the model
    model = NeuralNetwork(dummy_args)
    
    #Loading weights
    weights_dict = load_model(args.model_path)
    
    #Putting weights into network
    for i, layer in enumerate(model.layers):
        layer.W = weights_dict[f'W_{i}']
        layer.b = weights_dict[f'b_{i}']
        
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