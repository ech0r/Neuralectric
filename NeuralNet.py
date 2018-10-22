# General Neural Net framework
# Author: John Cooper (ech0r)
# Date: 9/29/2018

import math
import numpy as np

class NeuralNet:

    def __init__(self, input_size, output_size, hidden_size, num_hidden, bias):
        # array of outputs through network
        self.y = []
        self.weightedsum = []
        self.delta = []
        self.bias = np.zeros()
        # build array of weight matrices
        self.weights = []
        for i in range(num_hidden):
            if i == 0:
                self.weights.append(np.random.rand(input_size, hidden_size))
            elif i == (num_hidden-1):
                self.weights.append(np.random.rand(hidden_size, output_size))
            else:
                self.weights.append(np.random.rand(hidden_size,hidden_size))

    # TODO: implement more advanced activation functions like leaky ReLU
    def sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def sigmoid_d(self, x):
        return x*(1-x)

    def feed_forward(self, inputarray):
        for i in range(len(self.weights)):
            if i == 0:
                self.weightedsum.append(np.dot(inputarray,self.weights[i]))
                self.y.append(self.sigmoid(self.weightedsum[i]))
            else:
                self.weightedsum.append(np.dot(self.y[i-1], self.weights[i]))
                self.y.append(self.sigmoid(self.weightedsum[i]))

    def back_prop(self, trainingdata, expectedoutput):
        self.delta.append((self.y[-1] - expectedoutput)*self.sigmoid_d(self.weightedsum[-1])*self.y[-2])






    # TODO: write training function
    def train(self, input, expectedoutput, output):

    # TODO: write function to save weights after training

    # TODO: write function to predict based on new input

    # TODO: write main controller
















