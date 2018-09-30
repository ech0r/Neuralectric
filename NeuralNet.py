# General Neural Net framework
# Author: John Cooper (ech0r)
# Date: 9/29/2018

import math
import numpy as np

class NeuralNet:

    def __init__(self, input_size, output_size, hidden_size, num_hidden):

        self.num_hidden = num_hidden
        if self.num_hidden <= 1:
            self.num_hidden = 0
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # list to store hidden layer weights
        self.hidden_weights = []
        # weights from input layer to first hidden layer matrix input_size x hidden_size
        self.input_weight = np.random.randn(self.input_size, self.hidden_size)
        # for each hidden layer - generate weights
        for x in self.num_hidden:
            self.hidden_weights.append(np.random.randn(self.hidden_size, self.hidden_size))
        self.output_weight = np.random.randn(self.hidden_sizem, self.output_size)

    # TODO: implement more advanced activation functions
    def sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def sigmoid_d(self, x):
        return x*(1-x)

    def feed_forward(self, input):
        output = np.dot(input, self.input_weights)
        # first activation past the input layer
        activated_output = self.sigmoid(output)
        for x in self.hidden_weights:
            activated_output = np.dot(activated_output, x)
        output = np.dot(activated_output, self.output_weights)
        output = self.sigmoid(output)
        return output

    # TODO: write backpropagation function
    def back_prop(self,):
        print("do backprop stuff")

    # TODO: write training function

    # TODO: write function to save weights after training

    # TODO: write function to predict based on new input

    # TODO: write main controller
















