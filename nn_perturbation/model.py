from .params import Params
from .perturbations import Perturbations
from utils import *


class ModelPerturbation:
    def __init__(self, name, learning_rate):
        self.name = name
        self.params = Params()
        self.perturbations = None
        self.learning_rate = learning_rate

    def push_layer(self, output_dim, input_dim=None, f=relu):
        self.params.push_layer(output_dim, input_dim, f)

    def init_params(self, Ws, Bs):
        self.params.Ws = Ws
        self.params.Bs = Bs
        self.params.update_theta()

    def get_params(self):
        return self.params.Ws, self.params.Bs

    def init_perturbations(self, nb_params, dA):
        self.perturbations = Perturbations(nb_params, dA)

    def get_output(self, X):
        """
        X: (batch, in_dim)
        returns: (batch, out_dim)
        """
        for W, B, f in zip(self.params.Ws, self.params.Bs, self.params.fs):
            X = f(W @ X + B)  # (batch,out_dim)
        return X

    def get_p_output(self, X):
        """
        X: (input_dim, batch_size)
        returns: (T, output_dim, batch_size)
        """
        pThetas = self.perturbations.perturb(self.params)  # Shape: (T, nb_params)
        Ws, Bs = self.params.from_pThetas(pThetas)  # Ws: list of (T, out, in), Bs: list of (T , out, 1)

        for W, B, f in zip(Ws, Bs, self.params.fs):
            X = W @ X + B
            X = f(X)

        return X

    @staticmethod
    def get_loss(Y_pred, Y_true):
        return np.sum((Y_pred - Y_true) ** 2, axis=0)

    def gradient(self, pLoss, Loss):
        L_diff = pLoss - Loss
        Sum = np.einsum('i,ij->ij', L_diff, self.perturbations.Ps)
        return np.sum(Sum, axis=0) / self.perturbations.dA ** 2

    def do_descent(self, grad):
        self.params.Theta = self.params.Theta - self.learning_rate * grad
        self.params.update_weights_and_biases()

    def train(self, data_X, data_Y, do_print=False):
        if do_print:
            print("\n Model Type : Perturbation \n")

        nb_batches = data_X.shape[0]

        Losses = np.zeros([nb_batches])

        for batch_idx in range(nb_batches):
            X, Y_true = data_X[batch_idx], data_Y[batch_idx]  # (in, batch), (out, batch)
            Y_pred = self.get_output(X)  # (out, batch)
            batch_loss = np.mean(self.get_loss(Y_pred, Y_true))  # scalar

            Losses[batch_idx] = batch_loss

            # forward with perturbations
            pY_pred = self.get_p_output(X)  # (T, batch, out)
            pLoss = np.mean((pY_pred - Y_true[None, :, :]) ** 2, axis=(1, 2))  # (T, )

            # descent
            grad = self.gradient(pLoss, batch_loss)
            self.do_descent(grad)

            if do_print and batch_idx in nb_batches // 10 * np.arange(10):
                print(f"batch n° : {batch_idx}, loss : {Losses[batch_idx]}")

        return Losses

    def test(self, data):
        X_test, Y_true = data.X[0], data.Y[0]  # (in, n), (out, n)
        Y_pred = self.get_output(X_test)  # (out, n)

        loss = np.mean((Y_pred - Y_true) ** 2, axis=(0, 1))
        print(f"loss : {np.mean(loss)}")
        return Y_pred
