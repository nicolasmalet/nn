import matplotlib.pyplot as plt
from Trainer import train
from plot import plot_losses


def train_models(models, data, do_print=False, do_plot=False):
    Losses = []
    for model in models:
        Loss = train(model, data, 0.01, do_print)
        Losses.append(Loss)

    if do_plot:
        plot_losses(models, Losses, savepath=None)


def train2(models, data, do_print=False, do_plot=False):
    Losses = []
    for model in models:
        Loss = model.train(data, do_print)
        Losses.append(Loss)

    if do_plot:
        plot_losses(models, Losses, savepath=None)
