import numpy as np


class MSE:
    @staticmethod
    def get_avg_loss(Y_pred, Y_true):
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: avg loss shape: float
        """
        return np.mean((Y_pred - Y_true) ** 2, axis=(0, 1))

    @staticmethod
    def get_batch_loss(Y_pred, Y_true):
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        return np.mean((Y_pred - Y_true) ** 2, axis=0)

    @staticmethod
    def get_p_loss(pY_pred, Y_true):
        """
        :param pY_pred: (T, out, batch)
        :param Y_true: (out, batch)
        :return: perturbed loss (T, )
        """
        return np.mean((pY_pred - Y_true) ** 2, axis=(1, 2))

    @staticmethod
    def get_d_loss(Y_pred, Y_true):
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        return 2 * np.mean(Y_pred - Y_true, axis=0)


