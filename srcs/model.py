import numpy as np

class LinearRegression():

    def __init__(self, n_weights=1):
        self.weights = np.zeros(n_weights)
        self.bias = 0

    def load_parameters(self, file):
        pass

    def predict(self, inputs):
        return self.weights * inputs + self.bias

    def __str__(self):
        output = 'Weights = ' +  str(self.weights) + ', '
        output += 'Bias = ' + str(self.bias)
        return output