import numpy as np
from sklearn.datasets import fetch_openml
from .data import Data
mnist = fetch_openml('mnist_784', version=1)


X = np.array(mnist.data.astype("float32") / 255.0).T  # 70 000 images
Y = np.array(mnist.target.astype("int64")).reshape(1, -1)


X_train, X_test = X[:60000], X[60000:]
Y_train, Y_test = Y[:60000], Y[60000:]

data_mnist = Data(X_train, Y_train, X_test, Y_test, 20)


