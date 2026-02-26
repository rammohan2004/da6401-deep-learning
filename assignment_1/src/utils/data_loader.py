"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(dataset_name='mnist', val_split=0.1):
    """
    Loads, flattens, normalizes, and one-hot encodes the dataset.
    
    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
        val_split: Fraction of training data to use for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 1. Load raw data
    if dataset_name == 'mnist':
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()

    #flattening image from (N, 28, 28) to (N, 784)
    X_train_flat =X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test_flat =X_test_raw.reshape(X_test_raw.shape[0], -1)
    
    #Noemalizing to make values between 0.0 and 1.0
    X_train_norm = X_train_flat/255.0
    X_test_norm = X_test_flat/255.0

    #Doing one hot encoding for outputs
    y_train_encoded = np.eye(10)[y_train_raw]
    y_test_encoded = np.eye(10)[y_test_raw]
    
    #splitting training data into training and validaion
    X_train, X_val, y_train, y_val =train_test_split(X_train_norm, y_train_encoded, test_size=val_split)
    
    return X_train, X_val, X_test_norm, y_train, y_val, y_test_encoded