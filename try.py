import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from pandas.plotting import autocorrelation_plot

def left_to_right(origin, N):
    ret = origin.loc[N, range(1,21)]
    ret.index = pd.date_range('1975', periods=20, freq='AS')
    ret = pd.DataFrame(data=ret)
    ret.columns = ['values']
    return ret

def standard(origin):
    vals = origin.values
    min_max = preprocessing.MinMaxScaler()
    val_scaled = min_max.fit_transform(vals)
    ret = pd.DataFrame(val_scaled)
    ret.index = pd.date_range('1975', periods=20, freq='AS')
    ret.columns = ['values']
    return ret

def de_trend(origin):
    X = [i for i in range(0, len(origin))]
    X = np.reshape(X, (len(X), 1))
    print("org", origin)
    print("obs", observe)
    y = observe.values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)

    ret = pd.DataFrame(data=sig.detrend(origin['values']))
    ret.index = pd.date_range('1975', periods=20, freq='AS')
    ret.columns = ['values']
    return ret, trend

def stationary_test(df):
    rolmean = df.rolling(5).mean()
    rolstd = df.rolling(5).std()
    orig = plt.plot(df, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    result = adfuller(df['values'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

path = os.path.abspath("M3TrainingSet.xlsx")
xls = pd.ExcelFile(path)
original_data = pd.read_excel(xls)

#observe = original_data.loc[0:146,5]
#observe.index = original_data.loc[0:146,'Series'] These are for top to bottom

observe = left_to_right(original_data, 1) #select which series to investigate

print(observe.head()) #Checking that the right data is selected

scaled = standard(observe) #scaling the data

notrend, trend = de_trend(scaled) #detrending the data

print(trend)

stationary_test(observe) #checking if data is non-stationary (p-value>0.05)

#rolling_mean = observe.rolling(5).mean()
#df_log_minus_mean = observe - rolling_mean
#df_log_minus_mean.dropna(inplace=True)
#stationary_test(df_log_minus_mean)

stationary_test(notrend)

plt.plot(scaled, label='scaled')
plt.plot(notrend, label='detrended')
plt.title('Financial data starting in 1975')
plt.xlabel('Year', fontsize=10)
plt.legend()
plt.show()

autocorrelation_plot(notrend['values'])
plt.title('Autocorrelation plot')
plt.show()

result = seasonal_decompose(notrend, model='additive')
result.plot()
plt.show()

model = ARIMA(notrend, order=(2,1,0))
resultsar = model.fit(disp=-1)
plt.plot(notrend)
plt.plot(resultsar.fittedvalues, color='red')
plt.title('ARIMA fittedvalues')
plt.show()

resultsar.plot_predict(1,25)
plt.title('ARIMA predicted values')
plt.show()

print(resultsar.fittedvalues)
