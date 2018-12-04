import numpy as np
import matplotlib.pyplot as plt

#X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]),)
#Y = np.array(([0], [1], [1], [0]),)
#Y = np.array([0, 1, 1, 0])


class NeuralNet():
    def __init__(self, network_architecture, switch=None):
        # create seed for random number generation
        np.random.seed(0)
        self.switch = switch
        self.num_layers = len(network_architecture)
        self.architecture = network_architecture
        self.weights = []
        self.error = 0.0
        self.errorlist = []
        # initialize weight values
        for layer in range(self.num_layers - 1):
            weight = 2*np.random.rand(network_architecture[layer] + 1, network_architecture[layer+1]) - 1
            self.weights.append(weight)

    def relu(self, x):
        return np.where(x < 0, 0.01 * x, x)

    def relu_d(self, x):
        return np.where(x < 0, 0.01, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return np.multiply(x, 1.0-x)

    def tanh(self, x):
        return (1.0 - np.exp(-2 * x)) / (1.0 + np.exp(-2 * x))

    def tanh_d(self, x):
        return (1 + self.tanh(x)) * (1 - self.tanh(x))

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

    def forward(self, x):
        y = x
        for i in range(len(self.weights)-1):
            weighted_sum = np.dot(y[i], self.weights[i])
            layer_output = self.act(weighted_sum)

            # add bias - always on neuron
            layer_output = np.concatenate((np.ones(1), np.array(layer_output)))
            y.append(layer_output)
        weighted_sum = np.dot(y[-1], self.weights[-1])
        layer_output = self.act(weighted_sum)
        y.append(layer_output)
        return y

    def backward(self, y, known, learning_rate):
        error = known - y[-1]
        error_delta = [error * self.act_d(y[-1])]
        self.error = error
        # starting from 2nd to last layer
        for i in range(self.num_layers-2, 0, -1):
            error = error_delta[-1].dot(self.weights[i][1:].T)
            error = error*self.act_d(y[i][1:])
            error_delta.append(error)
        # we reverse the list of layer deltas to match the order of our weights
        error_delta.reverse()
        # now we update our weights using the delta from each layer
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, self.architecture[i]+1)
            delta = error_delta[i].reshape(1, self.architecture[i+1])
            self.weights[i] += learning_rate*layer.T.dot(delta)

    def train(self, data, labels, learning_rate=0.1, epochs=10000):
        # add bias to input layer - always on
        ones = np.ones((1, data.shape[0]))
        z = np.concatenate((ones.T, data), axis=1)
        for k in range(epochs):
            if (k+1) % 10000 == 0:
                print('epochs: {}'.format(k+1))
            sample = np.random.randint(X.shape[0])
            # feed data forward through our network
            x = [z[sample]]
            y = self.forward(x)

            known = labels[sample]
            self.backward(y, known, learning_rate)
            self.errorlist.append(self.error)

    def saveWeights(self):
        print("save weights")

    def predict(self, x):
        result = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0, len(self.weights)):
            result = self.act(np.dot(result, self.weights[i]))
            result = np.concatenate((np.ones(1).T, np.array(result)))
        return result[1]




np.random.seed(0)

NN = NeuralNet([6, 4, 1], "relu")

data = np.genfromtxt('load.csv', delimiter=',', skip_header=True)
X = data[:, :-1]
Y = data.T[-1]
# get maximum of Y and normalize Y - load numbers are too large
max = np.amax(Y)
Y = Y/max

NN.train(X, Y, learning_rate=0.01, epochs=1000000)


data = np.genfromtxt('test.csv', delimiter=',', skip_header=True)
input = data[:, :-1]
output = data.T[-1]

print("Final prediction")
totalerror = 0
hours = []
predicted_output = []
count = 0
for i in range(len(input)):
    count += 1
    hours.append(count)
    predicted = NN.predict(input[i])*max
    predicted_output.append(predicted)
    totalerror += (abs(output[i]-predicted)/output[i])*100
    #print(output[i], predicted, (abs(output[i]-predicted)/output[i])*100)
print("Average error", totalerror/len(input))

plt.plot(hours, output)
plt.plot(hours, predicted_output)
plt.show()

