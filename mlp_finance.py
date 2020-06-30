import numpy as np


class NeuralNetwork(object):

    #Constructor 
    def __init__(self):
        #size
        self.inputsize = 3
        self.hiddensize = 2 
        self.outputsize = 1 
        #weights 
        self.W1 = np.random.randn(self.inputsize, self.hiddensize) # 1 * 3 weight matrix
        self.W2 = np.random.randn(self.hiddensize, self.outputsize) # 3 * 1 weight matrix 

    # FF through the networks
    def feedForward(self, X):
        self.A = np.matmul(np.transpose(X), self.W1)#np.dot(X, self.W1) # Dotproduct, singlenumber 
        self.A2 = np.matmul(self.A, self.W2) # Dot product of hidden layer and second set of weights 
        return self.A2

    def derivative_MSE(self, Y, output):
        return output * 2 * (Y-output)

    def delta_W(self, A, t, learning_rate):
        return learning_rate * np.mean(self.derivative_MSE(A, t))

    # Backward propagate through the network (min. error, thus the sum of squares)
    def backward(self, X, Y, output):
        self.derivative_A2 = self.derivative_MSE(Y, output)
        self.derivative_A1 = self.delta_W(self.A2, output, 0.9)
        print(self.derivative_A2)
        print(X)
        self.W1 = np.transpose(X) * self.derivative_A1
        self.W2 = np.matmul(np.transpose(self.A2), self.derivative_A2) #self.z2.T.dot(self.output_delta)

    # Trains the FF
    def train(self, X, Y):
        output = self.feedForward(X)
        self.backward(X, Y, output)

    #returns a normalized array
    def normalizer(sef, data):
        norm = np.linalg.norm(data)
        return data/norm

if __name__ == '__main__':

    #training set 
    #X = [networth timestep N], Y = [networh timestep N + 1]
    X = np.array(([900.10], [948.48], [1020.21]), dtype = float)
    Y = np.array(([1040.24]), dtype = float) 

    neuralNetwork = NeuralNetwork()
    #X = neuralNetwork.normalizer(X)
    #Y = neuralNetwork.normalizer(Y)

    for i in range(1000):
        neuralNetwork.train(X, Y)

    print("predicted output: ", neuralNetwork.feedForward(X))
        
    