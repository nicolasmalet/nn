import numpy as np


def relu(x):
    return np.maximum(x, 0)


def heaviside(x):
    return np.heaviside(x, 0.5)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)



def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def identity(x):
    return x


def ones(x):
    return np.ones_like(x)


get_df = {
    relu: heaviside,
    sigmoid: sigmoid_derivative,
    identity: ones,
}


def MSE(Y_pred, Y_true):
    return -np.sum(Y_true * np.log(Y_pred))


def cross_entropy(Y_pred, Y_true):
    return -np.log(Y_pred[np.arange(Y_pred.shape[1]), Y_true])


def d_cross_entropy(Y_pred, Y_true):
    return -np.log(Y_pred[np.arange(Y_pred.shape[1]), Y_true])
