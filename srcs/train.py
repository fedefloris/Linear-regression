from model import LinearRegression
from optimizer import Optimizer
from dataset import Dataset
from loss import mse

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot(model, n_weights, losses, dataset):
    if n_weights == 1:
        _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        ax1.set_title('Predictions')
        ax1.plot(dataset.x_raw, dataset.y, 'x', color="#9376AC")
        ax1.plot(dataset.x_raw, model.predict(dataset.x), label='prediction', color="#172457")
        ax1.set_xlabel(dataset.x_label[0])
        ax1.set_ylabel(dataset.y_label)
        ax1.legend()
    else:
        _, ax2 = plt.subplots(ncols=1, figsize=(10, 5))
    ax2.plot(losses, label='mse', color="#172457")
    ax2.set_title('Loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('error')
    ax2.legend()
    plt.show()

def train(dataset, args):
    n_weights = np.size(dataset.x, 1)
    model = LinearRegression(n_weights=n_weights)
    optimizer = Optimizer(dataset)
    losses = []
    for _ in range(args.epochs):
        losses.append(mse(model, dataset))
        optimizer.step(model, args.learning_rate)
    print ('Model:', model, '\n')
    print ('Loss:', mse(model, dataset))
    sklearn_model = SklearnLinearRegression().fit(dataset.x, dataset.y)
    print ('Loss with sklearn:', mean_squared_error(dataset.y, sklearn_model.predict(dataset.x)), '\n')
    if args.plot:
        plot(model, n_weights, losses, dataset)
    return model

def get_args():
    parser = argparse.ArgumentParser(
        description='Train a linear regression model.'
    )
    parser.add_argument(
        '-dataset',
        default='data/cars.csv'
    )
    parser.add_argument(
        '-epochs',
        type=int,
        default=350
    )
    parser.add_argument(
        '-learning_rate',
        type=float,
        default=1.3
    )
    parser.add_argument(
        '-plot',
        action='store_true',
        help='Plot the predictions and the losses after training'
    )
    parser.add_argument(
        '-model_parameters',
        default='parameters.json',
        help='File that will contain the trained parameters of the model'
    )
    return parser.parse_args()

def main():
    args = get_args()
    dataset = Dataset(args.dataset)
    model = train(dataset, args)
    print ('Saving parameters to', args.model_parameters)
    model.save_parameters(dataset, args.model_parameters)

if __name__ == '__main__':
    main()
