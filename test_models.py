from Tester import test
from plot import plot_predictions


def test_models(models, data, do_plot=False):
    Y_preds = []
    for model in models:
        Y_pred = test(model, data)
        Y_preds.append(Y_pred)

    if do_plot:
        plot_predictions(models, data, Y_preds, savepath=None)
