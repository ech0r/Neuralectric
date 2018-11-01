# General Neural Net framework
# Author: John Cooper (ech0r)
# Date: 9/29/2018

import math
import numpy as np

class NeuralNet:

    def __init__(self, num_input, num_output, hidden_depth, num_hidden):
        # array of outputs through network
        self.y = []
        self.weightedsum = []
        self.delta = []
        # build array of weight matrices
        self.weights = []
        self.bias = []
        for i in range(num_hidden + 1):
            if i == 0:  # first layer - input to hidden
                self.bias.append(np.zeros((1, hidden_depth),))
                self.weights.append(np.random.rand(num_input, hidden_depth))
            elif i == num_hidden:  # last layer - hidden to output
                self.bias.append(np.zeros((1, num_output),))
                self.weights.append(np.random.rand(hidden_depth, num_output))
            else:
                self.bias.append(np.zeros((1, hidden_depth),))
                self.weights.append(np.random.rand(hidden_depth, hidden_depth))

    def relu(self, x):
        return np.where(x < 0, 0.01*x, x)

    def relu_d(self, x):
        return np.where(x > 0, 1.0, 0.0)

    def sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def sigmoid_d(self, x):
        return x*(1-x)

    def feed_forward(self, inputarray):
        self.y.append(inputarray)
        for i in range(len(self.weights)):
            if i == 0:
                self.weightedsum.append(np.dot(inputarray, self.weights[i]) + self.bias[i])
                self.y.append(self.relu(self.weightedsum[i]))
            else:
                self.weightedsum.append(np.dot(self.y[i-1], self.weights[i]) + self.bias[i])
                self.y.append(self.relu(self.weightedsum[i]))

    def back_prop(self, x, y, o):
        # get output layer error
        self.o_error = y - o
        self.o_delta = self.o_error*self.relu_d(o)
        for i,e in reversed(list(enumerate(self.weights))):
            layer_error = self.o_delta.dot(self.weights[i].T)
            layer_delta = layer_error*self.relu_d(self.y)

    def train(self, x_in, expectedoutput):
            self.feed_forward(x_in)
            self.back_prop(x_in, expectedoutput[i])

    def saveweights(self):
        for i in range(len(self.weights)):
            np.savetxt("w" + i + ".txt", self.weights[i], fmt="%s")

    def predict(self, input):
        self.feed_forward(input)

    def clear(self):
        self.y = []
        self.weightedsum = []

X = np.array([
    0, 0,
    0, 1,
    1, 0,
    1, 1
]).reshape(4, 2)

Y = np.array([0, 1, 1, 0]).reshape(4, 1)
NN = NeuralNet(2, 1, 3, 2)
NN.train(X,Y)
print(NN.weightedsum)
















