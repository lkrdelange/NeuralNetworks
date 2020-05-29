from program.Reader import *
from program.MathPlot import *
import tensorflow as tf


# Class that inits the NN
class NeuralNet(object):

    def __init__(self):
        self.file = Reader.readfile()
        self.observations = self.file.loc[0:645, 'Demographic'].values
        self.train_split = 515
        self.past_hist = 20
        self.future_target = 0
        self.batch_size = 256
        self.buffer_size = 10000
        tf.random.set_seed(13)
        self.calculus = MathPlot()
        self.scenario()

    def scenario(self):
        observe_train_mean = self.calculus.get_mean(self.observations, self.train_split)
        observe_train_std = self.calculus.get_std(self.observations, self.train_split)
        observe_standardized = self.calculus.get_standardized(self.observations, observe_train_mean, observe_train_std)

        x_train, y_train = self.calculus.univariate_data(observe_standardized, 0, self.train_split, self.past_hist,
                                                         self.future_target)
        x_val, y_val = self.calculus.univariate_data(observe_standardized, self.train_split, None, self.past_hist,
                                                     self.future_target)
        self.history(x_train, y_train)

        self.calculus.show_plot([x_train[0], y_train[0], self.calculus.baseline(x_train[0])], 0, 'baseline sample')

        self.neural_network(x_train, y_train, x_val, y_val)

    def neural_network(self, x_train, y_train, x_val, y_val):
        train_uni = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_uni = train_uni.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        val_uni = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_uni = val_uni.batch(self.batch_size).repeat()
        simple_lstm = tf.keras.models.Sequential(
            [tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]), tf.keras.layers.Dense(1)])
        simple_lstm.compile(optimizer='adam', loss='mae')
        EVALUATION_INTERVAL = 200
        EPOCHS = 10
        simple_lstm.fit(train_uni, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_uni,
                        validation_steps=50)
        for x, y in val_uni.take(3):
            self.calculus.show_plot([x[0].numpy(), y[0].numpy(), simple_lstm.predict(x)[0]], 0, 'Simple LSTM')

    @staticmethod
    def history(x_train, y_train):
        print('Single win of past')
        print(x_train[0])
        print('\n Target')
        print(y_train[0])
