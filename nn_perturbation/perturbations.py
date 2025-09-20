import numpy as np


class Perturbations:
    def __init__(self, nb_params, dA):
        self.nb_params = nb_params
        self.dA = dA
        self.T = self.nb_params

        self.Ps = self.dA * np.eye(self.nb_params)  # Perturb one parameter at a time

    def perturb(self, params):
        return params.Theta[None, :] + self.Ps

    def test_P(self):
        return np.round(self.Ps.T @ self.Ps / self.dA ** 2, 2)
