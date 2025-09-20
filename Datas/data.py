import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, batch_size):
        input_dim = X_train.shape[0]
        output_dim = Y_train.shape[0]
        self.batch_size = batch_size

        self.X_train = X_train.reshape(-1, input_dim, batch_size)
        self.Y_train = Y_train.reshape(-1, output_dim, batch_size)
        self.X_test = X_test.reshape(1, input_dim, -1)
        self.Y_test = Y_test.reshape(1, output_dim, -1)

        self.nb_batches = X_train.shape[0]

    def show_data(self):
        plt.scatter(self.X_test.flatten(), self.Y_test.flatten())






