from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, name, loss):
        self.name = name
        self.loss = loss

    @abstractmethod
    def push_layer(self, output_dim):
        return NotImplementedError

    @abstractmethod
    def init_params(self, Ws, Bs):
        return NotImplementedError

    @abstractmethod
    def get_params(self):
        return NotImplementedError

    @abstractmethod
    def get_output(self, X):
        return NotImplementedError

    @abstractmethod
    def show_weights(self):
        return NotImplementedError
