from dataset import Dataset
from model import LinearRegression
from loss import mse
from optimizer import Optimizer

import matplotlib.pyplot as plt
import argparse

def plot(model, losses, dataset):
    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Predictions')
    ax1.plot(dataset.x_raw, dataset.y, 'or')
    ax1.plot(dataset.x_raw, model.predict(dataset.x), label='prediction')
    ax1.set_xlabel('km')
    ax1.set_ylabel('price')
    ax1.legend()
    ax2.plot(losses, label='mse')
    ax2.set_title('Loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('error')
    ax2.legend()
    plt.show()

def train(dataset, args):
    model = LinearRegression()
    optimizer = Optimizer(dataset)
    losses = []
    for _ in range(args.epochs):
        losses.append(mse(model, dataset))
        optimizer.step(model)
    print ('Model:', model)
    print ('Loss:', mse(model, dataset))
    if args.plot:
        plot(model, losses, dataset)
    return model

def positive_int(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value

def get_args():
    parser = argparse.ArgumentParser(
        description='Train a linear regression model for car price prediction.'
    )
    parser.add_argument(
        '-dataset',
        default='data/cars.csv',
        help='CSV file containing car features (including the price)'
    )
    parser.add_argument(
        '-epochs',
        type=positive_int,
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
