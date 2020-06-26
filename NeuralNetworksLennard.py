import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path = os.path.abspath("M3TrainingSet.xlsx")
xls = pd.ExcelFile(path)

original_data = pd.read_excel(xls)#, usecols = "G:BA")
#print(original_data.head())
observe = original_data.iloc[1:147, 6:26]
observe.index = original_data.loc[1:147,'Series']
print(observe)
print(len(observe))

def calc_mean(dataset):
    diff = list()
    #dataset.plot(subplots = True)
    #plt.show()
    for i in range(1, len(dataset)):
        value = dataset.iloc[i] - dataset.iloc[i-1]
        diff.append(value)
    return diff
    #plt.plot(diff)
    #plt.show()
        
    
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])

    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length,0))

def show_plot(data, delta, title):
    labels = ['History', 'True Future', 'Model prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(data[0].shape[0])
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
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-step')
    plt.show()
    return plt

def baseline(history):
    return np.mean(history)

detrended = list ()
for timeseries in range(len(observe)):
    detrended.append(calc_mean(observe.iloc[timeseries]))

print(len(detrended))
TRAIN_SPLIT = 14
tf.random.set_seed(13)
print(detrended)#list with timeseries with 19 entries [1976:1994]
observe_train_mean = detrended[:TRAIN_SPLIT].mean()
observe_train_std = detrended[:TRAIN_SPLIT].std()

#observe = (observe-observe_train_mean)/observe_train_std

#past_hist = 20
#future_target = 0

#x_train, y_train = univariate_data(observe, 0, TRAIN_SPLIT, past_hist, future_target)

#x_val, y_val = univariate_data(observe, TRAIN_SPLIT, None, past_hist, future_target)

#print('Single win of past')
#print(x_train[0])
#print('\n Target')
#print(y_train[0])

#show_plot([x_train[0], y_train[0], baseline(x_train[0])], 0, 'baseline sample')

#BATCH_SIZE = 256
#BUFFER_SIZE = 10000

#train_uni = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_uni = train_uni.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_uni = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_uni = val_uni.batch(BATCH_SIZE).repeat()

simple_lstm = tf.keras.models.Sequential()#[tf.keras.layers.LSTM(4, input_shape=x_train.shape[-2:]), tf.keras.layers.Dense(1)])#units 8 -> 4
simple_lstm.add(tf.keras.layers.LSTM(8), input_shape=x_train.shape[-2:])
simple_lstm.add(tf.keras.layers.Dense(1))
simple_lstm.compile(optimizer='adam', loss='mae')

EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm.fit(train_uni, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_uni, validation_steps=50)

for x, y in val_uni.take(3):
  plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm.predict(x)[0]], 0, 'Simple LSTM model')
