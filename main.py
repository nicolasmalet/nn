from init_models import init_models
from train_models import train_models
from test_models import test_models
import matplotlib.pyplot as plt
from Datas import create_data_func, create_data_mnist


def main():
    data = create_data_func()
    models = init_models()
    train_models(models, data, do_print=True, do_plot=True)
    test_models(models, data, do_plot=True)
    plt.show()


if __name__ == "__main__":
    main()
