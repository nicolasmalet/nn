import numpy as np
from utils import relu


class Params:
    def __init__(self):
        self.Ws, self.Bs, self.fs = [], [], []
        self.W_shapes, self.B_shapes = [], []
        self.W_sizes, self.B_sizes = [], []
        self.nb_layers, self.nb_params = 0, 0
        self.Theta = np.array([])

    def push_layer(self, output_dim, input_dim=None, f=relu):
        input_dim = self.B_sizes[-1] if input_dim is None else input_dim
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        W = np.random.uniform(-limit, limit, (output_dim, input_dim))
        B = np.zeros((output_dim, 1))
        self.Ws.append(W)
        self.Bs.append(B)
        self.fs.append(f)
        self.W_shapes.append(W.shape)
        self.B_shapes.append(B.shape)
        self.W_sizes.append(W.size)
        self.B_sizes.append(B.size)
        self.nb_layers += 1
        self.nb_params += W.size + B.size
        self.update_theta()

    def update_theta(self):
        self.Theta = np.concatenate([W.ravel() for W in self.Ws] + [B.ravel() for B in self.Bs])

    def update_weights_and_biases(self):
        self.Ws, self.Bs = [], []
        idx = 0
        for size, shape in zip(self.W_sizes, self.W_shapes):
            self.Ws.append(self.Theta[idx:idx + size].reshape(shape))
            idx += size
        for size, shape in zip(self.B_sizes, self.B_shapes):
            self.Bs.append(self.Theta[idx:idx + size].reshape(shape))
            idx += size

    def from_pThetas(self, Thetas):
        if Thetas.ndim == 1:
            Thetas = Thetas[None, :]
        N = Thetas.shape[0]

        Ws, Bs = [], []
        idx = 0
        for size, shape in zip(self.W_sizes, self.W_shapes):
            Ws.append(Thetas[:, idx:idx + size].reshape(N, *shape))
            idx += size
        for size, shape in zip(self.B_sizes, self.B_shapes):
            Bs.append(Thetas[:, idx:idx + size].reshape(N, *shape))
            idx += size
        return Ws, Bs
