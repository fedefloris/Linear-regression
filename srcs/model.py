class LinearRegression():

    def __init__(self, n_weights = 1):
        self.weights = [0 for _ in range(n_weights)]
        self.bias = 0

    def load_parameters(self, file):
        pass

    def predict(self, inputs):
        prediction = self.bias
        for x, w in zip(inputs, self.weights):
            prediction += x * w
        return prediction

    def __str__(self):
        output = 'Weights = ' +  str(self.weights) + ', '
        output += 'Bias = ' + str(self.bias)
        return output