import matplotlib.pyplot as plt


def train(models, X_train, Y_train, do_print=False, do_plot=False):
    Losses = []
    for model in models:
        Loss = model.train(X_train, Y_train, do_print)
        Losses.append(Loss)

    if do_plot:
        fig, axs = plt.subplots(1, len(models), figsize=(12, 4))
        for i in range(len(models)):
            axs[i].loglog(range(len(Losses[i])), Losses[i])
            axs[i].set_title(f"Loss with : {models[i].name}")
