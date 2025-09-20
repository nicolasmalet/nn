import matplotlib.pyplot as plt


def test(models, X_test, Y_test, do_plot=False):
    Y_preds = []
    for model in models:
        Y_pred = model.test(X_test, Y_test)
        Y_preds.append(Y_pred)

    if do_plot:
        fig, axs = plt.subplots(1, len(models), figsize=(12, 4))
        for i in range(len(models)):
            axs[i].plot(X_test[0, 0], Y_test[0, 0], color="green")
            axs[i].plot(X_test[0, 0], Y_preds[i][0], color="blue")
            axs[i].set_title(f"Predictions with : {models[i].name}")
