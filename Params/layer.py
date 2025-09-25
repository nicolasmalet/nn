import numpy as np


class Layer:
    def __init__(self, output_dim, input_dim, f, df):
        self.output_dim = output_dim
        self.f = f
        self.df = df

        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.B = np.zeros((output_dim, 1))
        self.X = np.array([])
        self.Z = np.array([])
        self.A = np.array([])

    def get_output(self, X):
        self.X = X
        self.Z = np.matmul(self.W, X) + self.B
        self.A = self.f(self.Z)
        return self.A

    def get_gradient(self, dL_dA):
        df_Z = self.df(self.Z)
        dL_dZ = df_Z * dL_dA
        dL_dW = np.matmul(dL_dZ, self.X.T) / self.X.shape[1]
        dL_dB = np.mean(dL_dZ, axis=1, keepdims=True)
        dL_dA = np.matmul(self.W.T, dL_dZ)
        return dL_dA, dL_dW, dL_dB

    def do_backprop(self, dL_dA, learning_rate):
        dL_dA, dW, dB = self.get_gradient(dL_dA)
        self.W = self.W - learning_rate * dW
        self.B = self.B - learning_rate * dB
        return dL_dA
