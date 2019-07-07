from dataset import Dataset
from model import LinearRegression
from loss import mse, mae, mpe, mape

import matplotlib.pyplot as plt
import argparse

def evaluate(args):
    dataset = Dataset(args.dataset)
    model = LinearRegression()
    print ('Loading parameters from', args.model_parameters)
    model.load_parameters(args.model_parameters)
    print ('Model:', model)
    print ('Mean square error:', mse(model, dataset))
    print ('Mean absolute error:', mae(model, dataset))
    print ('Mean percentage error: ', mpe(model, dataset))
    print ('Mean absolute percentage error: ', mape(model, dataset))

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a linear regression model for car price prediction.'
    )
    parser.add_argument(
        '-dataset',
        default='data/cars.csv',
        help='CSV file containing car features (including the price)'
    )
    parser.add_argument(
        '-model_parameters',
        default='parameters.json',
        help='File that contains the trained parameters of the model'
    )
    return parser.parse_args()

def main():
    args = get_args()
    evaluate(args)

if __name__ == '__main__':
    main()
