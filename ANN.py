import numpy as np
import math
import pandas

x = np.array(([0, 0], [0, 1], [1, 0], [1, 1]),)
y = np.array(([0], [1], [1], [0]),)

class NeuralNet:

    def __init__(self, input_size, output_size, hidden_layer_size, num_hidden):
        self.layer_outputs = []
        self.layer_output_delta = []
        self.weights = []
        self.learningrate = 0.01
        for i in range(num_hidden + 1):
            if i == 0:
                self.weights.append(np.random.rand(input_size, hidden_layer_size))
            elif i == num_hidden:
                self.weights.append(np.random.rand(hidden_layer_size, output_size))
            else:
                self.weights.append(np.random.randn(hidden_layer_size, hidden_layer_size))

    def relu(self, x):
        return np.where(x < 0, 0.01*x, x)

    def relu_d(self, x):
        return np.where(x < 0, 0.01, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return x*(1-x)

    def clear(self):
        self.layer_outputs = []

    def feed_forward(self, x):
        for i, e in enumerate(list(self.weights)):
            if i == 0:
                self.layer_outputs.append(self.sigmoid(np.dot(x, self.weights[i])))
            else:
                self.layer_outputs.append(self.sigmoid(np.dot(self.layer_outputs[i-1], self.weights[i])))

    def back_prop(self, x, y):
        error = y - self.layer_outputs[-1]
        error_delta = error*self.sigmoid_d(self.layer_outputs[-1])
        self.layer_output_delta.append(error_delta)
        j = 0
        for i, e in reversed(list(enumerate(self.weights))):
            self.layer_output_delta.append(self.layer_output_delta[j]@e.T*self.sigmoid_d(self.layer_outputs[i-1]))
            j += 1
        self.layer_output_delta.reverse()
        for i in range(len(self.weights)):
            self.layer_output_delta[i].shape = (-1, 1)
            self.layer_outputs[i].shape = (-1, 1)
            if i == 0:
                print(self.weights[i].shape)
                print(x.shape)
                print(self.layer_output_delta[i].shape)
                #self.weights[i] += self.layer_output_delta[i]*x.T
            else:
                self.weights[i] += self.layer_output_delta[i]*self.layer_outputs[i].T


## Can calculate layer deltas, just need to update weights correctly now.  TODO: fix below code



Net = NeuralNet(2, 1, 3, 2)

Net.feed_forward(x[0])
Net.back_prop(x[0], y[0])
print(Net.layer_outputs)
print(Net.layer_output_delta)
print(Net.weights)