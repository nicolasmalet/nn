import numpy as np

from .layer import Layer
from utils import *


class ModelBackpropagation:
    def __init__(self, name, learning_rate):
        self.name = name
        self.learning_rate = learning_rate
        self.layers = []
        self.n = 0

    def push_layer(self, output_dim, input_dim=None, f=relu, df=None):
        self.layers.append(Layer(output_dim,
                                 self.layers[-1].n if input_dim is None else input_dim,
                                 f,
                                 get_df[f] if df is None else df))
        self.n += 1

    def init_params(self, Ws, Bs):
        for layer, W, B in zip(self.layers, Ws, Bs):
            layer.W = W
            layer.B = B

    def get_params(self):
        Ws, Bs = [], []
        for layer in self.layers:
            Ws.append(layer.W)
            Bs.append(layer.B)
        return Ws, Bs

    def get_output(self, X):
        for layer in self.layers:
            X = layer.get_output(X)
        return X

    def do_backprop(self, Y, Y_true):
        dL_dAl = self.get_dloss(Y, Y_true)
        for i in range(self.n - 1, -1, -1):
            dL_dAl = self.layers[i].do_backprop(dL_dAl, self.learning_rate)

    @staticmethod
    def get_loss(Y_pred, Y_true):
        return np.sum((Y_pred - Y_true) ** 2, axis=0)

    def get_loss(self, Y_pred, Y_true):
        return -np.sum(Y_true * np.log(Y_pred))

    @staticmethod
    def get_dloss(Y_pred, Y_true):
        return 2 * (Y_pred - Y_true)

    def show_weights(self):
        S = []
        for i in range(self.n):
            S.append(self.layers[i].show_weight())
        return np.array(S)

    def train(self, data_X, data_Y, do_print=False):
        if do_print:
            print(" \n Model Type : Backpropagation \n")

        nb_batches = data_X.shape[0]

        Losses = np.zeros([nb_batches])

        for batch_idx in range(nb_batches):
            X, Y_true = data_X[batch_idx], data_Y[batch_idx]  # (in, batch), (out, batch)
            Y_pred = self.get_output(X)  # (out, batch)

            batch_loss = np.mean(self.get_loss(Y_pred, Y_true))
            Losses[batch_idx] = batch_loss
            self.do_backprop(Y_pred, Y_true)

            if do_print and batch_idx in nb_batches // 10 * np.arange(10):
                print(f"batch n° : {batch_idx}, loss : {Losses[batch_idx]}")

        return Losses

    def test(self, data):
        X_test, Y_true = data.X[0], data.Y[0]  # (in, n), (out, n)
        Y_pred = self.get_output(X_test)  # (out, n)

        loss = self.get_loss(Y_pred, Y_true)

        print(f"loss : {np.mean(loss)}")

        return Y_pred
