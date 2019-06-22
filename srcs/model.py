import numpy as np

class LinearRegression():

    def __init__(self, n_weights):
        self.weights = np.zeros(n_weights)
        self.bias = 0

    def load_parameters(self, file):
        data = np.load(file)
        if len(data.shape) == 2:
            self.weights = data[0]
            self.bias = data[1]

    def save_parameters(self, file):
        np.save(file, [self.weights, self.bias])

    def predict(self, inputs):
        return self.weights * inputs + self.bias

    def __str__(self):
        output = 'Weights = ' +  str(self.weights) + ', '
        output += 'Bias = ' + str(self.bias)
        return output