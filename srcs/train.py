from model import LinearRegression
from dataset import Dataset

import matplotlib.pyplot as plt
import argparse
import numpy as np

np.set_printoptions(suppress=True)

def save(model, output_file):
    print (f'Saving to {output_file}:')

def plot(model, dataset):
    plt.title('Linear Regression')
    plt.plot(dataset.y, dataset.x, 'ob')
    plt.plot(model.predict(dataset.x), dataset.x)
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()

def loss(model, dataset):
    error = 0
    for features, output in zip(dataset.x, dataset.y):
        error += (model.predict(features) - output) ** 2
    return error / len(dataset.x)

def train(args):
    model = LinearRegression()
    dataset = Dataset(args.input_file)
    print ('Training...')
    m = len(dataset.x)
    for _ in range(1500):
        theta0 = 0
        theta1 = 0
        for features, output in zip(dataset.x, dataset.y):
            error = model.predict(features) - output
            theta0 += error
            theta1 += error * features[0]
        model.bias += -0.3 * theta0 / m
        model.weights[0] += -0.3 * theta1 / m
    plot(model, dataset)
    return model

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Train a linear regression model for car price prediction.'
    )
    parser.add_argument(
        'input_file',
        help='CSV file containing car features (including the price)'
    )
    parser.add_argument(
        '-output_file',
        default='parameters',
        help='Output file that will contain the trained parameters of the model'
    )
    return parser.parse_args()

def main():
    arguments = get_arguments()
    model = train(arguments)
    save(model, arguments.output_file)

if __name__ == '__main__':
    main()
