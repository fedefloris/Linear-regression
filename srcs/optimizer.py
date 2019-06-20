import numpy as np

class Optimizer():
    
    def __init__(self, dataset):
        self.dataset = dataset

    def step(self, model, learning_rate=3e-1):
        bias = 0
        weights = np.zeros(model.weights.shape)
        m = len(self.dataset.x)
        for features, output in zip(self.dataset.x, self.dataset.y):
            error = model.predict(features) - output
            bias += error
            weights += error * features
        model.bias -= learning_rate * bias / m
        model.weights -= learning_rate * weights / m