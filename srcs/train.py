from dataset import Dataset
from model import LinearRegression
from optimizer import Optimizer

import matplotlib.pyplot as plt
import argparse

def plot(model, losses, dataset):
    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Predictions')
    ax1.plot(dataset.y, dataset.x, 'or')
    ax1.plot(model.predict(dataset.x), dataset.x, label='prediction')
    ax1.set_xlabel('km')
    ax1.set_ylabel('price')
    ax1.legend()
    ax2.plot(losses, label='loss')
    ax2.set_title('Losses')
    ax2.set_xlabel('error')
    ax2.set_ylabel('epochs')
    ax2.legend()
    plt.show()

def loss(model, dataset):
    error = (model.predict(dataset.x) - dataset.y) ** 2
    return error.sum() / len(dataset.x)

def train(args):
    dataset = Dataset(args.dataset)
    model = LinearRegression(n_weights=1)
    optimizer = Optimizer(dataset)
    losses = []
    for _ in range(args.epochs):
        losses.append(loss(model, dataset))
        optimizer.step(model)
    print ('Model:', model)
    print ('Loss:', loss(model, dataset))
    if args.plot:
        plot(model, losses, dataset)
    return model

def get_args():
    parser = argparse.ArgumentParser(
        description='Train a linear regression model for car price prediction.'
    )
    parser.add_argument(
        'dataset',
        help='CSV file containing car features (including the price)'
    )
    parser.add_argument(
        '-epochs',
        type=int,
        default=300,
        help='Number of epochs'
    )
    parser.add_argument(
        '-plot',
        action='store_true',
        help='Plot the predictions and the losses after training'
    )
    parser.add_argument(
        '-model_parameters',
        default='parameters.npy',
        help='File that will contain the trained parameters of the model'
    )
    return parser.parse_args()

def main():
    args = get_args()
    model = train(args)
    print ('Saving parameters to', args.model_parameters)
    model.save_parameters(args.model_parameters)

if __name__ == '__main__':
    main()
