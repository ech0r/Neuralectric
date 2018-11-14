import numpy as np

X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]),)
Y = np.array(([0], [1], [1], [0]),)

class NeuralNet:

    def __init__(self, input_size, output_size, hidden_layer_size, num_hidden, switch=None):
        self.switch = switch
        self.layer_outputs = []
        self.layer_output_delta = []
        self.weightedsum = []
        self.weights = []
        self.learningrate = 0.1
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

    def act(self, x):
        if self.switch == "relu":
            return self.relu(x)
        else:
            return self.sigmoid(x)

    def act_d(self, x):
        if self.switch == "relu":
            return self.relu_d(x)
        else:
            return self.sigmoid_d(x)

    def clear(self):
        self.layer_outputs = []
        self.weightedsum = []
        self.layer_output_delta = []

    def feed_forward(self, x):
        self.layer_outputs.append(np.atleast_2d(x))
        self.weightedsum.append(np.atleast_2d(x))
        for z, e in enumerate(list(self.weights)):
            self.weightedsum.append(np.atleast_2d(self.layer_outputs[z]@self.weights[z]))
            self.layer_outputs.append(np.atleast_2d(self.act(self.weightedsum[-1])))

    def back_prop(self, x, y):
        error = np.atleast_2d(y - self.layer_outputs[-1])
        error_delta = np.atleast_2d(error*self.act_d(self.weightedsum[-1]))
        for z, e in enumerate(reversed(list(self.weights))):
            if z == 0:
                self.layer_output_delta.append(error_delta)
            else:
                self.layer_output_delta.append(self.layer_output_delta[z-1]@self.weights[-z].T*self.act_d(self.weightedsum[-(z+1)]))
        self.layer_output_delta.reverse()
        for k in range(len(self.weights)):
            self.weights[k] -= self.learningrate*self.layer_outputs[k].T@self.layer_output_delta[k]

Net = NeuralNet(2, 1, 2, 1)


for i in range(100):
    Net.feed_forward(np.atleast_2d(X))
    Net.back_prop(np.atleast_2d(X), np.atleast_2d(Y))
    for h in Net.layer_outputs:
        print(h.shape)
    Net.clear()

Net.feed_forward(np.atleast_2d(X))
print(Net.layer_outputs[-1])
