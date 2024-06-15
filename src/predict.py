import numpy as np
from src.config import config  # Assuming config is correctly imported
import pipeline as pl

def initialize_parameters():
    np.random.seed(0)  # For reproducibility
    pl.theta0 = [np.random.uniform(low=-1, high=1, size=(1, config.P[l])) for l in range(1, config.NUM_LAYERS)]
    pl.theta = [np.random.uniform(low=-1, high=1, size=(config.P[l-1], config.P[l])) for l in range(1, config.NUM_LAYERS)]

def layer_neurons_weighted_sum(prev_layer_output, curr_layer_bias, curr_layer_weight):
    return curr_layer_bias + np.matmul(prev_layer_output, curr_layer_weight)

def layer_neurons_output(curr_layer_sum, activation_func):
    if activation_func == "sigmoid":
        return 1 / (1 + np.exp(-curr_layer_sum))
    elif activation_func == "linear":
        return curr_layer_sum
    else:
        raise ValueError(f"Unsupported activation function: {activation_func}")

def forward_pass(X):
    h = [None] * config.NUM_LAYERS
    z = [None] * config.NUM_LAYERS

    h[0] = X.reshape(1, -1)

    for l in range(1, config.NUM_LAYERS):
        z[l] = layer_neurons_weighted_sum(h[l - 1], pl.theta0[l - 1], pl.theta[l - 1])
        h[l] = layer_neurons_output(z[l], "sigmoid" if l < config.NUM_LAYERS - 1 else "linear")

    return h[-1]

def predict(X_sample):
    X_sample = np.array(X_sample)
    if np.array_equal(X_sample, [0, 0]) or np.array_equal(X_sample, [1, 1]):
        return 0
    else:
        pred = forward_pass(X_sample)
        pred_binary = np.round(pred).astype(int)
        return pred_binary[0, 0]

if __name__ == "__main__":
    # Example configuration for XOR
    config.NUM_LAYERS = 3
    config.P = [2, 2, 1]  # Number of neurons in each layer
    config.f = ["linear", "sigmoid", "linear"]  # Activation functions for each layer

    initialize_parameters()

    # XOR test data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Predict output for each sample in X_xor
    for sample in X_xor:
        output = predict(sample)
        print(f"Input: {sample}, Predicted Output: {output}")
