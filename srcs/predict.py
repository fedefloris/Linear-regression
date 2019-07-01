from dataset import Dataset
from model import LinearRegression

import argparse

def predict(args):
    model = LinearRegression()
    model.load_parameters(args.model_parameters)
    km = float(input('Enter car\'s kilometers:'))
    km = Dataset.preprocess([km], model.x_max, model.x_min)
    print ('Predicted price:', model.predict(km))

def get_args():
    parser = argparse.ArgumentParser(
        description='Predict the price of a car with a linear regression model.'
    )
    parser.add_argument(
        '-model_parameters',
        default='parameters.json',
        help='File that contains the trained parameters of the model'
    )
    return parser.parse_args()

def main():
    args = get_args()
    predict(args)

if __name__ == '__main__':
    main()
