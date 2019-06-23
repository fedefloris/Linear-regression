from dataset import Dataset
from model import LinearRegression

import matplotlib.pyplot as plt
import argparse

def plot(model, dataset):
    pass

def evaluate(args):
    dataset = Dataset(args.dataset)
    model = LinearRegression()
    model.load_parameters(args.model_parameters)
    print (model)
    plot(model, dataset)

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a linear regression model for car price prediction.'
    )
    parser.add_argument(
        'dataset',
        help='CSV file containing car features (including the price)'
    )
    parser.add_argument(
        '-model_parameters',
        default='parameters.npy',
        help='File that contains the trained parameters of the model'
    )
    return parser.parse_args()

def main():
    args = get_args()
    evaluate(args)

if __name__ == '__main__':
    main()
