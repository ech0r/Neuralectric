# General Neural Net framework
# Author: John Cooper (ech0r)
# Date: 9/29/2018

import math
import numpy as np

class NeuralNet:

    def __init__(self, input_size, output_size, hidden_size, num_hidden):
        self.z = 0
        self.z_array = []
        self.activated_z = 0
        self.num_hidden = num_hidden - 1
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
        self.output_weight = np.random.randn(self.hidden_size, self.output_size)

    # TODO: implement more advanced activation functions like ReLU
    def sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def sigmoid_d(self, x):
        return x*(1-x)

    def feed_forward(self, input):
        i = 0
        for x in range(self.num_hidden + 2):
            if x == 0:  # input layer -> first hidden layer
                self.z = np.dot(input, self.input_weight)
                self.activated_z = self.sigmoid(self.z)
                self.z_array.append(self.activated_z)
            elif x == (self.num_hidden + 1):  # final hidden layer -> output
                self.z = np.dot(self.activated_z, self.output_weight)
                self.activated_z = self.sigmoid(self.z)
                self.z_array.append(self.activated_z)
                return self.activated_z
            else:  # hidden layer -> hidden layer
                self.z = np.dot(self.activated_z, self.hidden_weights[i])
                self.activated_z = self.sigmoid(self.z)
                self.z_array.append(self.activated_z)
                i += 1

    def back_prop(self, traininginputarray, expectedoutputarray, outputarray):
        # subtract NN output from known value
        ax = 0
        mse = ((expectedoutputarray - outputarray))**2).mean(axis=ax)
        # get activated error
        error_d = error*self.sigmoid_d(outputarray)
        # TODO: need to figure out how to correctly adjust n hidden layer's weights
        i = 0
        for x in range(self.num_hidden + 2):
            if x == 0:  # input -> hidden
                z_error = error_d.dot(self.input_weight.T)
                z_d = z_error*self.sigmoid_d(self.z_array[x])
                self.z_array[x] += traininginputarray.T.dot(z_d)
            elif x == self.num_hidden + 1:  # hidden -> output
                z_error = error_d.dot(self.output_weight.T)
                z_d = z_error*self.sigmoid_d(self.z_array[x])
                self.z_array[x] += traininginputarray.T.dot(z_d)
            else:  # hidden -> hidden
                z_error = error_d.dot(self.z_array[x].T)
                z_d = z_error*self.sigmoid_d(self.z_array[x])
                self.z_array[x] +=

    # TODO: write training function

    # TODO: write function to save weights after training

    # TODO: write function to predict based on new input

    # TODO: write main controller
















