from init_models import init_models
from train_models import train
from test_models import test
import matplotlib.pyplot as plt
from Datas import data_func, data_mnist


data = data_mnist

models = init_models()
train(models, data.X_train, data.Y_train, do_print=True, do_plot=True)
test(models, data.X_test, data.Y_test, do_plot=True)
plt.show()
