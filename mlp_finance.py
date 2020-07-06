import numpy as np


class NeuralNetwork(object):

	#Constructor 
	def __init__(self):
		self.inputsize = 1
		self.outputsize = 1 
		self.hiddensize = 3 
		#weights 
		self.W1 = np.random.randn(self.inputsize, self.hiddensize) # 1 * 3 weight matrix
		self.W2 = np.random.randn(self.hiddensize, self.outputsize) # 3 * 1 weight matrix 

	# FF through the network
	def feedForward(self, X):
		self.z = np.dot(X, self.W1) # Dotproduct, singlenumber 
		self.z2 = self.sigmoid(self.z) # Activiation function 
		self.z3 = np.dot(self.z2, self.W2) # Dot product of hidden layer and second set of weights 
		return self.sigmoid(self.z3)

	def sigmoid(self, s, deriv = False):
		if deriv == True:
			return s * (1 - s)
		return 1/(1 + np.exp(-s))

	# Backward propagate through the network (min. error, thus the sum of squares)
	def backward(self, X, Y, output):
		self.output_error = Y - output 
		self.output_delta = self.output_error * self.sigmoid(output, deriv = True)

		self.z2_error = self.output_delta.dot(self.W2.T) # Error, how much our hidden layer contributes to the output error
		self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv = True) # applying deriv. of sigmoid to z2 error 

		self.W1 = X.T.dot(self.z2_delta) #adjusting first set of weights (input -> hidden)
		self.w2 = self.z2.T.dot(self.output_delta)

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
	Y = np.array(([920.22], [960.98], [1040.24]), dtype = float) 

	neuralNetwork = NeuralNetwork()
	#X = neuralNetwork.normalizer(X)
	#Y = neuralNetwork.normalizer(Y)

	for i in range(1000):
		neuralNetwork.train(X, Y)

	print("predicted output: ", neuralNetwork.feedForward(X))
		
	