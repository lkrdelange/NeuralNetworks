import numpy as np


class NeuralNetwork(object):

    #Constructor
    def __init__(self):
        #size
        self.inputsize = 3
        self.hiddensize= 2
        self.outputsize = 1
        #weights
        self.W1_2 = np.random.randn(self.inputsize, self.hiddensize) # input to hidden weights
        self.W2_3 = np.random.randn(self.hiddensize, self.hiddensize) # hidden to hidden layer weights
        self.W3_4 = np.random.randn(self.hiddensize, self.outputsize) # hidden to output layer weights
        #bias
        self.bias = 1.0
        #learning rate
        self.learning_rate = 0.9
        #deltas
        #self.delta_hidden_output = 0.0
        #self.delta_hidden_hidden = 0.0
        #self.delta_input_hidden = 0.0
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
            return a * (1 - a)
        return 1/(1 + np.exp(-a))

    # FF through the networks
    def feedForward(self, X):
        self.potential_2 = np.matmul(X, self.W1_2)
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
        print("Biem", self.A3)
        delta_output = self.delta_hidden_output(estimated_output, Y)
        gradient_output = self.gradient_hidden_output(delta_output) # W3_4 has shape 2 * 1
        delta_hidden_L3 = self.delta_hidden_hidden(delta_output) # W2_3 has shape 2 * 2
        gradient_hidden_L3 = self.gradient_hidden_hidden(delta_hidden_L3)
        delta_hidden_L2 = self.delta_input_hidden(delta_hidden_L3) # W1_2 has shape 3 * 2
        gradient_hidden_L2 = self.gradient_input_hidden(delta_hidden_L2, X)
        self.W3_4 = self.W3_4 - (self.learning_rate * np.asarray(gradient_output))
        print("pre change w23",self.W2_3.shape)
        self.W2_3 = self.W2_3 - (self.learning_rate * np.asarray(gradient_hidden_L3))
        print("post change w23",self.W2_3.shape)

        self.W1_2 = self.W1_2 - (self.learning_rate * np.asarray(gradient_hidden_L2))
        #print("post change w12",self.W1_2.shape)


    # Trains the NN through back_propagation
    def back_propagation(self, X, Y):
        estimated_output = self.feedForward(X)
        return self.backward_pass(X, Y, estimated_output)

if __name__ == '__main__':

    X = np.array(([900.10], [948.48], [1020.21]), dtype = float).T
    Y = np.array(([1040.24]), dtype = float)

    neuralNetwork = NeuralNetwork()

    gradient = 0.0
    for i in range(1): #Epochs
        for  j in range(5): #From i to max length training set where i is one timeseries in data set
            neuralNetwork.back_propagation(X, Y)
        # emp_risk = (1 / 1000) * gradient
        # gradient = 0.0
        # print("[", i ,"]","Empericial risk is: ", emp_risk)

    print("predicted output: ", neuralNetwork.feedForward(X))
