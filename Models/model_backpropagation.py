from .model import Model
from Params.layer import Layer
from utils import *


class ModelBackpropagation(Model):
    def __init__(self, name, loss, learning_rate):
        super().__init__(name, loss)
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

    def do_descent(self, Y, Y_true):
        dL_dAl = self.loss.get_d_loss(Y, Y_true)
        for i in range(self.n - 1, -1, -1):
            dL_dAl = self.layers[i].do_backprop(dL_dAl, self.learning_rate)

    def show_weights(self):
        Ws, Bs = self.get_params()
        return f"Ws, {Ws}, Bs {Bs}"
