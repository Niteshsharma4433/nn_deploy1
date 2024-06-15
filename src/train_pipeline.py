import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model

import pipeline as pl

# Initialize variables
z = [None] * config.NUM_LAYERS
h = [None] * config.NUM_LAYERS

del_fl_by_del_z = [None] * config.NUM_LAYERS
del_hl_by_del_theta0 = [None] * config.NUM_LAYERS
del_hl_by_del_theta = [None] * config.NUM_LAYERS
del_L_by_del_h = [None] * config.NUM_LAYERS
del_L_by_del_theta0 = [None] * config.NUM_LAYERS
del_L_by_del_theta = [None] * config.NUM_LAYERS

def layer_neurons_weighted_sum(prev_layer_output, curr_layer_bias, curr_layer_weight):
    return curr_layer_bias + np.matmul(prev_layer_output, curr_layer_weight)

def layer_neurons_output(curr_layer_sum, activation_func):
    if activation_func == "linear":
        return curr_layer_sum
    elif activation_func == "sigmoid":
        return 1 / (1 + np.exp(-curr_layer_sum))
    elif activation_func == "tanh":
        return np.tanh(curr_layer_sum)
    elif activation_func == "relu":
        return np.maximum(0, curr_layer_sum)

def del_layer_neurons_outputs_wrt_weighted_sums(activation_func, curr_layer_sum):
    if activation_func == "linear":
        return np.ones_like(curr_layer_sum)
    elif activation_func == "sigmoid":
        curr_layer_output = 1 / (1 + np.exp(-curr_layer_sum))
        return curr_layer_output * (1 - curr_layer_output)
    elif activation_func == "tanh":
        return 1 - np.tanh(curr_layer_sum)**2
    elif activation_func == "relu":
        return (curr_layer_sum > 0).astype(float)

def del_layer_neurons_outputs_wrt_biases(curr_layer_dels):
    return curr_layer_dels

def del_layer_neurons_outputs_wrt_weights(prev_layer_output, curr_layer_dels):
    return np.matmul(prev_layer_output.T, curr_layer_dels)

def run_training(tol, epsilon, batch_size=2):
    epoch_counter = 0
    mse = 1
    loss_per_epoch = [mse]

    training_data = load_dataset("train.csv")

    preprocessor = pp.preprocess_data()
    preprocessor.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = preprocessor.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    pl.initialize_parameters()

    while True:
        mse = 0
        permutation = np.random.permutation(X_train.shape[0])
        X_train = X_train[permutation]
        Y_train = Y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            batch_mse = 0
            batch_grads_theta0 = [np.zeros_like(theta0) for theta0 in pl.theta0]
            batch_grads_theta = [np.zeros_like(theta) for theta in pl.theta]

            for j in range(X_batch.shape[0]):
                h[0] = X_batch[j].reshape(1, -1)

                for l in range(1, config.NUM_LAYERS):
                    z[l] = layer_neurons_weighted_sum(h[l - 1], pl.theta0[l], pl.theta[l])
                    h[l] = layer_neurons_output(z[l], config.f[l])
                    del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l], z[l])
                    del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
                    del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l - 1], del_fl_by_del_z[l])

                Y_batch[j] = Y_batch[j].reshape(-1, 1)
                L = 0.5 * (Y_batch[j][0] - h[config.NUM_LAYERS - 1][0, 0])**2
                batch_mse += L

                del_L_by_del_h[config.NUM_LAYERS - 1] = (h[config.NUM_LAYERS - 1] - Y_batch[j])

                for l in range(config.NUM_LAYERS - 2, 0, -1):
                    del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l + 1], (del_fl_by_del_z[l + 1] * pl.theta[l + 1]).T)

                for l in range(1, config.NUM_LAYERS):
                    batch_grads_theta0[l] += del_L_by_del_h[l] * del_hl_by_del_theta0[l]
                    batch_grads_theta[l] += del_L_by_del_h[l] * del_hl_by_del_theta[l]

            for l in range(1, config.NUM_LAYERS):
                pl.theta0[l] -= epsilon * (batch_grads_theta0[l] / batch_size)
                pl.theta[l] -= epsilon * (batch_grads_theta[l] / batch_size)

            mse += batch_mse / batch_size

        mse /= (X_train.shape[0] / batch_size)
        epoch_counter += 1
        loss_per_epoch.append(mse)

        print(f"Epoch # {epoch_counter}, Loss = {mse}")

        if abs(loss_per_epoch[-1] - loss_per_epoch[-2]) < tol:
            break

if __name__ == "__main__":
    run_training(1e-8, 1e-7)
    save_model(pl.theta0, pl.theta)
