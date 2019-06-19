from dataset import Dataset
from model import LinearRegression
from optimizer import Optimizer

import matplotlib.pyplot as plt
import argparse
import numpy as np

np.set_printoptions(suppress=True)

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
    dataset = Dataset(args.input_file)
    model = LinearRegression(n_weights=1)
    optimizer = Optimizer(dataset)
    for _ in range(args.epochs):
        optimizer.step(model)
        print (model)
    plot(model, dataset)
    return model

def get_args():
    parser = argparse.ArgumentParser(
        description='Train a linear regression model for car price prediction.'
    )
    parser.add_argument(
        '-epochs',
        type=int,
        default=100,
        help='Number of epochs'
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
    args = get_args()
    model = train(args)
    model.save_parameters(args.output_file)

if __name__ == '__main__':
    main()
