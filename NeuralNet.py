# General Neural Net framework
# Author: John Cooper (ech0r)
# Date: 9/29/2018

import math
import numpy as np

class NeuralNet:

    def __init__(self, input_size, output_size, hidden_size, num_hidden):
        # array of outputs through network
        self.y = []
        self.weightedsum = []
        self.delta = []
        # build array of weight matrices
        self.weights = []
        for i in range(num_hidden):
            if i == 0:
                self.weights.append(np.random.rand(input_size, hidden_size))
            elif i == (num_hidden-1):
                self.weights.append(np.random.rand(hidden_size, output_size))
            else:
                self.weights.append(np.random.rand(hidden_size, hidden_size))
        self.bias = np.zeros(len(self.weights))

    def relu(self, x):
        return np.where(x < 0, 0.01*x, x)

    def relu_d(self, x):
        return np.where(x > 0, 1.0, 0.0)

    def sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def sigmoid_d(self, x):
        return x*(1-x)

    def feed_forward(self, inputarray):
        for i in range(len(self.weights)):
            if i == 0:
                self.weightedsum.append(np.dot(inputarray, self.weights[i]) + self.bias[i])
                self.y.append(self.relu(self.weightedsum[i]))
            else:
                self.weightedsum.append(np.dot(self.y[i-1], self.weights[i]) + self.bias[i])
                self.y.append(self.relu(self.weightedsum[i]))

    def back_prop(self, input, expectedoutput):
        # get output layer error
        layer_error = []
        layer_error.append((self.y[-1] - expectedoutput)*self.relu_d(self.weightedsum[-1]))
        for i in range(len(self.weights)-1):
            layer_error.append(self.weights[-(i+1)].T*self.relu_d(self.weightedsum[-(i+2)]))

        for i in range(len(self.y)-1):
            # back at the beginning
            if i == 0:
                self.delta.append(layer_error[i]*self.y[-(i+2)])
            elif i == len(self.y)-1:
                self.delta.append(layer_error[i-1]*layer_error[i]*input)
            else:
                self.delta.append(layer_error[i-1]*layer_error[i]*self.y[-(i+2)])


    # TODO: write training function
    def train(self, input, expectedoutput):
        self.feed_forward(input)
        self.back_prop(input, expectedoutput)

    # TODO: write function to save weights after training
    def saveweights(self):
        for i in range(len(self.weights)):
            np.savetxt("w" + i + ".txt", self.weights[i], fmt="%s")

    def predict(self, input):
        self.feed_forward(input)

X = np.array([
    0, 0,
    0, 1,
    1, 0,
    1, 1
]).reshape(4,2)

Y = np.array([0, 1, 1, 0]).reshape(4, 1)
NN = NeuralNet(2, 1, 3, 2)
#print(NN.weights)
#print(NN.bias)
for j in range(1000):
    for x in range(len(X)):
        NN.train(X, Y)

NN.predict(X[2])
print(NN.y[-1])

















