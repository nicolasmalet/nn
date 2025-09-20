from nn_backpropagation import ModelBackpropagation
from nn_perturbation import ModelPerturbation
from utils import *


def init_models():
    models = [ModelBackpropagation(name="Backprop", learning_rate=100)]
              #ModelPerturbation(name="Perturb", learning_rate=1e-1)]

    for model in models:
        model.push_layer(input_dim=28**2, output_dim=256, f=relu)
        model.push_layer(input_dim=256, output_dim=256, f=relu)
        model.push_layer(input_dim=256, output_dim=10, f=softmax)

    Ws, Bs = models[0].get_params()
    for model in models:
        model.init_params(Ws, Bs)
        if isinstance(model, ModelPerturbation):
            model.init_perturbations(model.params.nb_params, 1e-10)

    return models
