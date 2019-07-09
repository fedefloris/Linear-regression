import numpy as np

class Optimizer():
    
    def __init__(self, dataset):
        self.dataset = dataset

    def step(self, model, learning_rate):
        m = len(self.dataset.y)
        errors = model.predict(self.dataset.x) - self.dataset.y
        bias = errors.sum() / m
        weights = np.dot(self.dataset.x.T, errors) / m
        model.bias -= learning_rate * bias
        model.weights -= learning_rate * weights