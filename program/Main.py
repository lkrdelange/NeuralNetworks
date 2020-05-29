from program.NeuralNet import *


class Main(object):

    if __name__ == '__main__':
        NeuralNet();


path = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(path, 'M3Forecast.xls')
original_data = pd.read_excel(file, 'SINGLE')
print(original_data.head())

observe = original_data.loc[0:645,'Demographic']
observe.index = original_data.loc[0:645,'N']
print(observe.head())

#observe.plot(subplots = True)
#plt.show()



TRAIN_SPLIT = 515
tf.random.set_seed(13)

observe = observe.values
observe_train_mean = observe[:TRAIN_SPLIT].mean()
observe_train_std = observe[:TRAIN_SPLIT].std()

observe = (observe-observe_train_mean)/observe_train_std

past_hist = 20
future_target = 0

x_train, y_train = univariate_data(observe, 0, TRAIN_SPLIT, past_hist, future_target)

x_val, y_val = univariate_data(observe, TRAIN_SPLIT, None, past_hist, future_target)

print('Single win of past')
print(x_train[0])
print('\n Target')
print(y_train[0])

show_plot([x_train[0], y_train[0], baseline(x_train[0])], 0, 'baseline sample')

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_uni = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_uni = train_uni.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_uni = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_uni = val_uni.batch(BATCH_SIZE).repeat()

simple_lstm = tf.keras.models.Sequential([tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]), tf.keras.layers.Dense(1)])
simple_lstm.compile(optimizer='adam', loss='mae')

EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm.fit(train_uni, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_uni, validation_steps=50)

for x, y in val_uni.take(3):
  plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm.predict(x)[0]], 0, 'Simple LSTM model')
