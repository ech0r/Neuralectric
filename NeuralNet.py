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
        self.bias = []
        # build array of weight matrices
        self.weights = []
        for i in range(num_hidden):
            if i == 0:
                self.weights.append(np.random.rand(input_size, hidden_size))
                self.bias.append(np.zeros((input_size, hidden_size)))
            elif i == (num_hidden-1):
                self.weights.append(np.random.rand(hidden_size, output_size))
                self.bias.append(np.zeros((hidden_size, output_size)))
            else:
                self.weights.append(np.random.rand(hidden_size, hidden_size))
                self.bias.append(np.zeros((hidden_size, hidden_size)))

    def relu(self, x):
        if x < 0:
            return 0.01*x
        else:
            return x

    def relu_d(self, x):
        if x < 0:
            return 0.01
        else:
            return 1

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
        output_layer = (self.y[-1] - expectedoutput)*self.relu_d(self.weightedsum[-1])
        for x, e in reversed(list(enumerate(self.y))):
            z_error = output_layer.dot(self.weights[x].T)
            z_delta = z_error*self.relu_d(e)
            if i == 0:
                self.weights[x] += input.T.dot(z_delta)
            else:
                self.weights[x] += e.T.dot(output_layer)

    # TODO: write training function
    def train(self, input, expectedoutput):
        self.feed_forward(input)
        self.back_prop(input, expectedoutput, )

    # TODO: write function to save weights after training
    def saveweights(self):
        for i in range(len(self.weights)):
            np.savetxt("w" + i + ".txt", self.weights[i], fmt="%s")

X = np.array([
    0, 0,
    0, 1,
    1, 0,
    1, 1
]).reshape(4,2)

Y = np.array([0, 1, 1, 0]).reshape(4, 1)
NN = NeuralNet(2, 1, 3, 2)
print(NN.weights)
print(NN.bias)
# for j in range(1000):
#    for i in range(len(X)):
        #NN.train(X, Y)


















