import numpy as np


class LinearLayer:
    def __init__(self, size, input_size):
        self.weights = np.random.randn(size, input_size)
        self.biases = np.random.rand(size, 1)

        self.last_x = np.zeros((input_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x) -> np.matrix:
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # x is an Nx1
    def forward(self, x):
        self.last_x = x
        return self.sigmoid(self.weights.dot(x) + self.biases)

    # loss_dy is an Nx1 matrix of loss from previous layer, M vectors of size N
    # x is an Mx1 matrix of M = input_size
    def backprop(self, loss_dy: np.matrix):
        # h is Nx1
        h = self.weights.dot(self.last_x) + self.biases

        grad = loss_dy * self.sigmoid_gradient(h)

        # dW must be NxM
        dW = np.matmul(grad, np.transpose(self.last_x))
        dB = grad

        # update weights & biases
        self.weights = self.weights - dW * 0.001
        self.biases = self.biases - dB * 0.001

        dY = np.matmul(np.transpose(self.weights), grad)

        return dY
