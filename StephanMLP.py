import numpy as np
import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from pandas.plotting import autocorrelation_plot
from scipy.special import expit
import scipy.signal as sig
import matplotlib.pyplot as plt



class NeuralNetwork(object):

    #Constructor
    def __init__(self):
        #size
        self.inputsize = 3
        self.hiddensize= 2
        self.outputsize = 1
        #weights
        self.W1_2 = np.random.randn(self.inputsize, self.hiddensize) # input to hidden weights
        np.clip(self.W1_2, -3, 3)
        self.W2_3 = np.random.randn(self.hiddensize, self.hiddensize) # hidden to hidden layer weights
        np.clip(self.W2_3, -3, 3)
        self.W3_4 = np.random.randn(self.hiddensize, self.outputsize) # hidden to output layer weights
        np.clip(self.W3_4, -3, 3)

        #bias
        self.bias = 1.0
        #learning rate
        self.learning_rate = 0.2
        #potential
        self.potential_2 = 0.0
        self.potential_3 = 0.0
        #Activation
        self.A2 = 0.0
        self.A3 = 0.0
        self.A4 = 0.0


    #Activation function
    def sigmoid(self, a, deriv=False):
        if (deriv == True):
            #print("a;",a)
            return a * (1 - a)
        return expit(a)

    # FF through the networks
    def feedForward(self, X):
        #print("X",X)
        self.potential_2 = np.matmul(X, self.W1_2)
        #print("selfpot2",self.potential_2)
        self.A2 = self.sigmoid(self.potential_2 + self.bias)
        self.potential_3 = np.matmul(self.A2, self.W2_3)
        self.A3 = self.sigmoid(self.potential_3 + self.bias)
        self.A4 = np.matmul(self.A3, self.W3_4) + self.bias
        return self.A4

    #Delta for output unit
    def delta_hidden_output(self, estimated_output, Y):
        return 2 * (estimated_output - Y)

    #Gradient for output unit
    def gradient_hidden_output(self, delta):
        gradient1 = delta * self.A3[0][0]
        gradient2 = delta * self.A3[0][1]
        return [[gradient1[0][0]], [gradient2[0][0]]]

    #Delta for L3 units
    def delta_hidden_hidden(self, delta_output):
        summation_left_node = delta_output * self.W3_4[0]
        summation_right_node = delta_output * self.W3_4[1]
        return [[self.sigmoid(self.potential_3[0][0], deriv=True) * summation_left_node], [self.sigmoid(self.potential_3[0][1], deriv=True) * summation_right_node]]

    #Gradient for L3 units
    def gradient_hidden_hidden(self, delta):
        gradient1 = delta[0][0] * self.A2[0][0]
        gradient2 = delta[0][0] * self.A2[0][1]
        gradient3 = delta[1][0] * self.A2[0][0]
        gradient4 = delta[1][0] * self.A2[0][1]
        return [gradient1[0][0], gradient2[0][0]],[ gradient3[0][0], gradient4[0][0]]

    #Delta for L2 units
    def delta_input_hidden(self, delta_hidden_L3):
        summation_left_node = delta_hidden_L3[0][0] * (self.W2_3[0][0] + self.W2_3[0][1])
        summation_right_node = delta_hidden_L3[1][0] * (self.W2_3[0][1] + self.W2_3[1][1])
        return [[self.sigmoid(self.potential_2[0][0], deriv=True) * summation_left_node], [self.sigmoid(self.potential_2[0][1], deriv=True) * summation_right_node]]

    #Gradient for L2 units
    def gradient_input_hidden(self, delta, X):
        gradient1 = delta[0][0] * X[0][0]
        gradient2 = delta[0][0] * X[0][1]
        gradient3 = delta[0][0] * X[0][2]
        gradient4 = delta[1][0] * X[0][0]
        gradient5 = delta[1][0] * X[0][1]
        gradient6 = delta[1][0] * X[0][2]
        return [gradient1[0][0], gradient2[0][0]], [gradient3[0][0] , gradient4[0][0]], [gradient5[0][0], gradient6[0][0]]

    # Backward propagate through the network (min. error, thus the sum of squares)
    def backward_pass(self, X, Y, estimated_output):
        #print("estim:",estimated_output,"Y:",Y)
        delta_output = self.delta_hidden_output(estimated_output, Y)
        #print("delata_output:",delta_output)
        gradient_output = self.gradient_hidden_output(delta_output) # W3_4 has shape 2 * 1
        delta_hidden_L3 = self.delta_hidden_hidden(delta_output) # W2_3 has shape 2 * 2
        gradient_hidden_L3 = self.gradient_hidden_hidden(delta_hidden_L3)
        delta_hidden_L2 = self.delta_input_hidden(delta_hidden_L3) # W1_2 has shape 3 * 2
        #print("deltaL2:",delta_hidden_L2)
        gradient_hidden_L2 = self.gradient_input_hidden(delta_hidden_L2, X)
        #print("l2gradient:", gradient_hidden_L2)
        self.W3_4 = self.W3_4 - (self.learning_rate * np.asarray(gradient_output))
        self.W3_4 = np.clip(self.W3_4, -2, 2)
        self.W2_3 = self.W2_3 - (self.learning_rate * np.asarray(gradient_hidden_L3))
        self.W2_3 = np.clip(self.W2_3, -2, 2)
        self.W1_2 = self.W1_2 - (self.learning_rate * np.asarray(gradient_hidden_L2))
        self.W1_2 = np.clip(self.W1_2, -2, 2)

        #print("weight12:",self.W1_2)


    # Trains the NN through back_propagation
    def back_propagation(self, X, Y):
        estimated_output = self.feedForward(X)
        #print("estim:",estimated_output)
        return self.backward_pass(X, Y, estimated_output)

def left_to_right(origin, N):
    ret = origin.loc[N, range(1,21)]
    ret.index = pd.date_range('1975', periods=20, freq='AS')
    ret = pd.DataFrame(data=ret)
    ret.columns = ['values']
    return ret

def standardize(origin):
    vals = origin.values
    min_max = preprocessing.MinMaxScaler()
    val_scaled = min_max.fit_transform(vals)
    ret = pd.DataFrame(val_scaled)
    ret.index = pd.date_range('1975', periods=20, freq='AS')
    ret.columns = ['values']
    return ret

def de_trend(origin, index):
    X = [i for i in range(0, len(origin))]
    X = np.reshape(X, (len(X), 1))
    #print("org", origin)
    #print("obs", observe[0])
    y = observe[index].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)

    ret = pd.DataFrame(data=sig.detrend(origin['values']))
    ret.index = pd.date_range('1975', periods=20, freq='AS')
    ret.columns = ['values']
    return ret#, trend

if __name__ == '__main__':


    path = os.path.abspath("M3TrainingSet.xlsx")
    xls = pd.ExcelFile(path)
    original_data = pd.read_excel(xls)

    j=0#because [0] contains years etc.
    observe = []
    while j < 146:
        observe.append(left_to_right(original_data, j))
        j+=1

    j=0
    scaled = []
    while j < 146:
        scaled.append(standardize(observe[j]))
        j+=1

    j=0
    notrend = []
    while j < 146:
        notrend.append(de_trend(scaled[j],j))
        j+=1



    X = []
    Y = []

    # print(len(notrend))
    # print(notrend[0])
    i = 0
    while i < 146:
        j = 0
        # x_3 = np.array((notrend[i].iloc[13:16]), dtype= float)
        # y_1 = np.array((notrend[i].iloc[16]), dtype= float)
        # X.append(x_3)
        # Y.append(y_1)
        while j < 20:
            x_3 = np.array((notrend[i].iloc[j:j+3]), dtype= float)
            y_1 = np.array((notrend[i].iloc[j+3]), dtype= float)
            X.append(x_3)
            Y.append(y_1)
            j= j+4
        i+=1
    #print("y2",Y[13])
    #print("lenghts X:",len(X),"Y",len(Y))

    #Xnep = np.array(([900.10], [948.48], [1020.21]), dtype = float).T
    #Ynep = np.array(([1040.24]), dtype = float)

    #neuralNetwork = NeuralNetwork()
    #k=0
    #pretrainingprediction = []
    #for k in range(len(X)):
    #    pretrainingprediction.append(neuralNetwork.feedForward(X[k].T))

    #print("predicted output: ", neuralNetwork.feedForward(X[k].T))
    p=0
    meanFit = []
    while p < 100:
        neuralNetwork = NeuralNetwork()

        gradient = 0.0
        j=0
        while j < len(X):
        #for  j in range(len(X)): #From i to max length training set where i is one timeseries in data set
            neuralNetwork.back_propagation(X[j].T, Y[j])
            j+=1
            neuralNetwork.back_propagation(X[j].T, Y[j])
            j+=1
            neuralNetwork.back_propagation(X[j].T, Y[j])
            j=j+3

        # emp_risk = (1 / 1000) * gradient
        # gradient = 0.0
        # print("[", i ,"]","Empericial risk is: ", emp_risk)

        fitness = []
        i=0
        j=0
        while i < len(X):
            if j == 99:
                iEnd = i + 3
                prediction1 = neuralNetwork.feedForward(X[iEnd].T)
                X2 = []
                X2.append(X[iEnd][1])
                X2.append(X[iEnd][2])
                X2.append(prediction1)
                prediction2 = neuralNetwork.feedForward(np.array(X2,dtype=float).T)

                X3 = []
                X3.append(X[iEnd][2])
                X3.append(prediction1)
                X3.append(prediction2)
                prediction3 = neuralNetwork.feedForward(np.array(X3,dtype=float).T)

                X4 = []
                X4.append(prediction1)
                X4.append(prediction2)
                X4.append(prediction3)
                prediction4 = neuralNetwork.feedForward(np.array(X4,dtype=float).T)

                X5 = []
                X5.append(prediction2)
                X5.append(prediction3)
                X5.append(prediction4)
                prediction5 = neuralNetwork.feedForward(np.array(X5,dtype=float).T)

                #
                plt.plot(notrend[j].values, 'b', label='True data')
                # print("notrend15",notrend[j].values[15])
                # print("pred15",prediction1[0])
                fitness.append(notrend[j].values[15] - prediction1[0])
                fitness.append(notrend[j].values[16] - prediction2[0])
                fitness.append(notrend[j].values[17] - prediction3[0])
                fitness.append(notrend[j].values[18] - prediction4[0])
                fitness.append(notrend[j].values[19] - prediction5[0])

                predictionline = notrend[j].values
                predictionline[15]=prediction1[0]
                predictionline[16]=prediction2[0]
                predictionline[17]=prediction3[0]
                predictionline[18]=prediction4[0]
                predictionline[19]=prediction5[0]

                plt.ylim(-1,1)
                plt.plot(predictionline, 'r', label='Prediction')
                #plt.plot(notrend[j].values, 'r', label='True values')
                plt.title('Financial forecast, 2 hidden layers, learning rate = 0.2')
                plt.xlabel('Year', fontsize=10)
                plt.legend()
                plt.show()
            i=i+5
            j+=1
        arr = np.asarray(fitness)
        meanFit.append(arr.mean())
        print("fitness",arr.mean())
        p+=1
    meanFitnp = np.asarray(meanFit)
    print("Final results, mean: ", meanFitnp.mean(), " Std dev: ", meanFitnp.std())
    plt.ylim(-0.5,0.5)
    plt.plot(meanFit, 'r', label='Fitness')
    plt.title('Fitness over 100 trials, 2 hidden layers, learning rate = 0.05\n Mean = '+str(round(meanFitnp.mean(),6))+', Standard deviation = '+str(round(meanFitnp.std(),6)))
    plt.ylabel('Fitness (mean difference between prediction and true values)')
    plt.xlabel('Trials', fontsize=10)
    plt.legend()
    plt.show()
