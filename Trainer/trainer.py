import numpy as np
from Models import ModelPerturbation
from Models import ModelBackpropagation


def train(model, data, learning_rate, do_print=False):
    if isinstance(model, ModelPerturbation):
        return train_model_perturbation(model, data, learning_rate, do_print)
    if isinstance(model, ModelBackpropagation):
        return train_model_backpropagation(model, data, learning_rate, do_print)


def train_model_perturbation(model, data, learning_rate, do_print):
    if do_print:
        print("Model Type : Perturbation \n")

    nb_batches = data.X_train.shape[0]

    Losses = np.zeros([nb_batches])

    for batch_idx in range(nb_batches):
        X, Y_true = data.X_train[batch_idx], data.Y_train[batch_idx]
        Y_pred = model.get_output(X)
        avg_loss = model.loss.get_avg_loss(Y_pred, Y_true)

        Losses[batch_idx] = avg_loss

        # forward with perturbations
        pY_pred = model.get_p_output(X)
        pLoss = model.loss.get_p_loss(pY_pred, Y_true)

        # descent
        grad = model.gradient(pLoss, avg_loss)
        model.do_descent(grad)

        if do_print and batch_idx in nb_batches // 10 * np.arange(10):
            print(f"batch n° : {batch_idx}, loss : {Losses[batch_idx]}")
    if do_print:
        print("\n")

    return Losses


def train_model_backpropagation(model, data, learning_rate, do_print):
    if do_print:
        print("Model Type : Backpropagation \n")

    nb_batches = data.X_train.shape[0]

    Losses = np.zeros([nb_batches])

    for batch_idx in range(nb_batches):
        X, Y_true = data.X_train[batch_idx], data.Y_train[batch_idx]  # (in, batch), (out, batch)
        Y_pred = model.get_output(X)  # (out, batch)

        avg_loss = model.loss.get_avg_loss(Y_pred, Y_true)
        Losses[batch_idx] = avg_loss
        model.do_descent(Y_pred, Y_true)

        if do_print and batch_idx in nb_batches // 10 * np.arange(10):
            print(f"batch n° : {batch_idx}, loss : {Losses[batch_idx]}")
    if do_print:
        print("\n")

    return Losses
