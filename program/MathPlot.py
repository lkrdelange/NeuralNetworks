import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Class containing the math for the NN
class MathPlot(object):

    @staticmethod
    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size

        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + target_size])

        return np.array(data), np.array(labels)

    @staticmethod
    def create_time_steps(length):
        return list(range(-length, 0))

    def show_plot(self, data, delta, title):
        labels = ['History', 'True Future', 'Model prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = self.create_time_steps(data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(data):
            if i:
                plt.plot(future, data[i], marker[i], markersize=10, label=labels[i])
            else:
                plt.plot(time_steps, data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel('Time-step')
        plt.show()
        return plt

    @staticmethod
    def baseline(history):
        return np.mean(history)

    @staticmethod
    def get_mean(data, split):
        return data[:split].mean()

    @staticmethod
    def get_std(data, split):
        return data[:split].std()

    @staticmethod
    def get_standardized(data, mean, std):
        return (data - mean) / std