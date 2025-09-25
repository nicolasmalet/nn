import numpy as np
from .data import Data


def create_data_func():
    nb_train = 10**5
    batch_size = 20
    nb_test = 1000

    a, b = - np.pi, np.pi
    f = np.sin

    U1 = np.random.uniform(a, b, [1, nb_train]).astype(np.float32)
    U2 = np.sort(np.random.uniform(a, b, [1, nb_test]).astype(np.float32), axis=1)
    data_func = Data(U1, f(U1), U2, f(U2), batch_size)

    return data_func
