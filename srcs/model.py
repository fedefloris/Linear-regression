import numpy as np
import json

class LinearRegression():

    def __init__(self, n_weights=1):
        self.weights = np.zeros((n_weights, 1))
        self.bias = 0

    def load_parameters(self, path):
        with open (path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if 'weights' in data and 'bias' in data:
                self.weights = np.array(data['weights'])
                self.bias = np.array(data['bias'])
            if 'x_min' in data and 'x_max' in data:
                self.x_min = np.array(data['x_min'])
                self.x_max = np.array(data['x_max'])

    def save_parameters(self, dataset, path):
        with open (path, 'w', encoding='utf-8') as file:
            data = {
                "weights": self.weights.tolist(),
                "bias": self.bias,
                "x_min": dataset.x_min.tolist(),
                "x_max": dataset.x_max.tolist(),
            }
            json.dump(data, file, indent=4)

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def __str__(self):
        output = 'Weights = ' +  str(self.weights.flatten()) + ', '
        output += 'Bias = ' + str(self.bias)
        return output