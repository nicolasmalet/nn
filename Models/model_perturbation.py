from .model import Model
from Params.params import Params
from optimizer.perturbations import Perturbations
from utils import *


class ModelPerturbation(Model):
    def __init__(self, name, loss, learning_rate):
        super().__init__(name, loss)
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

    def show_weights(self):
        return f"Ws, {self.params.Ws}, Bs {self.params.Bs}"

