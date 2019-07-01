import numpy as np

class Optimizer():
    
    def __init__(self, dataset):
        self.dataset = dataset

    def step(self, model, learning_rate=3e-1):
        bias = 0
        weights = np.zeros(model.weights.shape)
        m = len(self.dataset.y)
        errors = model.predict(self.dataset.x) - self.dataset.y
        bias = errors.sum() / m
        weights = (errors * self.dataset.x).sum() / m
        model.bias -= learning_rate * bias
        model.weights -= learning_rate * weights