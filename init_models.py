from Models import ModelBackpropagation
from Models import ModelPerturbation
from Loss import MSE
from utils import *


def init_models():
    models = [ModelBackpropagation(name="Backpropagation Model", loss=MSE, learning_rate=1e-1),
              ModelPerturbation(name="Perturbation Model", loss=MSE, learning_rate=1e-1)]

    for model in models:
        model.push_layer(input_dim=1, output_dim=10, f=relu)
        model.push_layer(input_dim=10, output_dim=10, f=relu)
        model.push_layer(input_dim=10, output_dim=1, f=identity)

    Ws, Bs = models[0].get_params()
    for model in models:
        model.init_params(Ws, Bs)
        if isinstance(model, ModelPerturbation):
            model.init_perturbations(model.params.nb_params, 1e-10)

    return models
