import numpy as np

X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]),)
Y = np.array(([0], [1], [1], [0]),)


class NeuralNet():
    def __init__(self, switch=None):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.switch = switch
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    def relu(self, x):
        return np.where(x < 0, 0.01 * x, x)

    def relu_d(self, x):
        return np.where(x < 0, 0.01, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return x * (1.0 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_d(self, x):
        return 1 - x * x

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

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.act(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.act(self.z3) # final activation function
        return o

    def backward(self, X, y, o):
        # backward propagate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error*self.act_d(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta@self.W2.T  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.act_d(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T@self.z2_delta  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T@self.o_delta  # adjusting second set (hidden --> output) weights

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self, X):
        print("Predicted data based on trained weights: ")
        print("Input: \n" + str(X))
        print("Output: \n" + str(self.forward(X)))


NN = NeuralNet("sigmoid")
for i in range(100000):  # trains the NN 1,000 times
    NN.train(X, Y)

NN.predict(X)
