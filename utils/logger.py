import logging
import os

from matplotlib import pyplot as plt


class Logger:
    def __init__(self):
        self.loss_train = []
        self.loss_val = []

        self.acc_train = []
        self.acc_val = []

    def get_logs(self):
        return self.loss_train, self.loss_val, self.acc_train, self.acc_val

    def restore_logs(self, logs):
        self.loss_train, self.loss_val, self.acc_train, self.acc_val = logs

    def save_plt(self, hps):
        loss_path = os.path.join(hps['model_save_dir'], 'loss.jpg')
        acc_path = os.path.join(hps['model_save_dir'], 'accuracy.jpg')

        plt.figure()
        plt.plot(self.acc_train, 'g', label='Treniranje')
        plt.plot(self.acc_val, 'b', label='Validacija')
        plt.title('Preciznost tijekom treniranja')
        plt.xlabel('Epoha')
        plt.ylabel('Preciznost')
        plt.legend()
        plt.grid()
        plt.savefig(acc_path)

        plt.figure()
        plt.plot(self.loss_train, 'g', label='Treniranje')
        plt.plot(self.loss_val, 'b', label='Validacija')
        plt.title('Gubitak tijekom treniranja')
        plt.xlabel('Epoha')
        plt.ylabel('Gubitak')
        plt.legend()
        plt.grid()
        plt.savefig(loss_path)
