import numpy as np

X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]),)
Y = np.array(([0], [1], [1], [0]),)

class NeuralNet:

    def __init__(self, hidden_layer_size, num_hidden, switch=None):
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden = num_hidden
        self.switch = switch
        self.layer_outputs = []
        self.layer_output_delta = []
        self.weightedsum = []
        self.weights = []
        self.bias = []
        self.num_hidden = num_hidden
        self.learningrate = 0.01

    # initialize weights to input dimensions
    def initialize_weights(self, x, y):
        input_size = x.shape[1]
        output_size = y.shape[1]
        for j in range(self.num_hidden + 1):
            if j == 0:
                #self.weights.append(np.random.uniform(low=0.0, high=0.5, size=(input_size, self.hidden_layer_size)))
                self.weights.append(np.random.randn(input_size, self.hidden_layer_size))
            elif j == self.num_hidden:
                #self.weights.append(np.random.uniform(low=-, high=0.5, size=(self.hidden_layer_size, output_size)))
                self.weights.append(np.random.randn(self.hidden_layer_size, output_size))
            else:
                #self.weights.append(np.random.uniform(low=-0.5, high=0.5, size=(self.hidden_layer_size, self.hidden_layer_size)))
                self.weights.append(np.random.randn(self.hidden_layer_size, self.hidden_layer_size))

    def relu(self, x):
        return np.where(x < 0, 0.01*x, x)

    def relu_d(self, x):
        return np.where(x < 0, 0.01, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return x * (1.0 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_d(self, x):
        return 1 - x*x

    def act(self, x):
        if self.switch == "relu":
            return self.relu(x)
        elif self.switch == "tanh":
            return self.tanh(x)
        else:
            return self.sigmoid(x)

    def act_d(self, x):
        if self.switch == "relu":
            return self.relu_d(x)
        elif self.switch == "tanh":
            return self.tanh(x)
        else:
            return self.sigmoid_d(x)

    def clear(self):
        self.layer_outputs = []
        self.weightedsum = []
        self.layer_output_delta = []

    def feed_forward(self, x):
        self.layer_outputs.append(self.act(np.atleast_2d(x)))
        self.weightedsum.append(np.atleast_2d(x))
        for z, e in enumerate(list(self.weights)):
            self.weightedsum.append(np.atleast_2d(np.dot(self.layer_outputs[z], self.weights[z])))
            self.layer_outputs.append(np.atleast_2d(self.act(self.weightedsum[-1])))

    def back_prop(self, y):
        error = np.atleast_2d(y - self.layer_outputs[-1])
        error_delta = np.atleast_2d(error * self.act_d(self.layer_outputs[-1]))
        for k in range(len(self.weights)):
            if k == 0:
                self.layer_output_delta.append(error_delta)
            else:
                weighted_error = np.dot(self.layer_output_delta[-1], self.weights[-k].T)*self.act_d(self.weightedsum[-(k+1)])
                self.layer_output_delta.append(weighted_error)
        self.layer_output_delta.reverse()
        for k in range(len(self.weights)):
            derivative = np.dot(self.layer_outputs[k].T, self.layer_output_delta[k])
            self.weights[k] += derivative


Net = NeuralNet(3, 1, "")
Net.initialize_weights(X, Y)
print(Net.weights)

for i in range(10000):
    Net.feed_forward(np.atleast_2d(X))
    Net.back_prop(np.atleast_2d(Y))
    Net.clear()

Net.feed_forward(np.atleast_2d(X))
print(Net.layer_outputs[-1])

