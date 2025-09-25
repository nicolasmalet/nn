import matplotlib.pyplot as plt


def plot_losses(models, Losses, savepath=None):
    """Plot training losses for each model."""
    fig, axs = plt.subplots(1, len(models), figsize=(12, 4), sharey=True)

    if len(models) == 1:
        axs = [axs]

    for i, model in enumerate(models):
        axs[i].loglog(range(len(Losses[i])), Losses[i],
                      color="C1", linewidth=1.5)
        axs[i].set_title(f"Loss with : {model.name}")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Loss")
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_predictions(models, data, Y_preds, savepath=None):
    """Plot predictions vs ground truth for each model."""
    fig, axs = plt.subplots(1, len(models), figsize=(12, 4), sharex=True, sharey=True)

    if len(models) == 1:
        axs = [axs]  # ensure iterable

    for i, model in enumerate(models):
        axs[i].plot(data.X_test[0, 0], data.Y_test[0, 0], color="green", linewidth=1.5, label="Ground truth")
        axs[i].plot(data.X_test[0, 0], Y_preds[i][0], color="blue", linewidth=1.5, label="Prediction")
        axs[i].set_title(f"Predictions with : {model.name}")
        axs[i].legend(fontsize=10)
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


