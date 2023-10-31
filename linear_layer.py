import numpy as np


class LinearLayer:
    def __init__(self, size, input_size, activation="relu"):
        self.weights = np.random.randn(size, input_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((size, 1))
        self.activation = activation

        self.last_x = np.zeros((input_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.clip(x, 0, None)

    def relu_gradient(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid_gradient(self, x) -> np.matrix:
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # x is an Nx1
    def forward(self, x):
        self.last_x = np.copy(x)
        h = self.weights.dot(x) + self.biases

        if self.activation == "sigmoid":
            return self.sigmoid(h)
        elif self.activation == "relu":
            return self.relu(h)
        else:
            assert False, f"unknown activation {self.activation}"

    # loss_dy is an Nx1 matrix of loss from previous layer, M vectors of size N
    def backprop(self, loss_dy: np.matrix):
        # h is Nx1
        if self.activation == "sigmoid":
            h = self.sigmoid_gradient(self.weights.dot(self.last_x) + self.biases)
            grad = loss_dy * h
        elif self.activation == "relu":
            h = self.relu_gradient(self.weights.dot(self.last_x) + self.biases)
            grad = loss_dy * h
        else:
            assert False, f"unknown activation {self.activation}"

        # dW must be NxM
        dW = np.matmul(grad, np.transpose(self.last_x))
        dB = grad

        # update weights & biases
        self.weights = self.weights - dW * 0.0001
        self.biases = self.biases - dB * 0.0001

        dY = np.matmul(np.transpose(self.weights), grad)

        return dY
