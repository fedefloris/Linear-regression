from dataset import Dataset
from model import LinearRegression

import numpy as np
import argparse

def predict(args):
    model = LinearRegression()
    print ('Loading parameters from', args.model_parameters, '\n')
    model.load_parameters(args.model_parameters)
    print ('Model:', model, '\n')
    x = []
    for index in range(len(model.weights)):
        x.append(float(input(f'Enter feature n.{index + 1}: ')))
    x = Dataset.preprocess(x, model.x_max, model.x_min)
    print ('Predicted value:', model.predict(x))

def get_args():
    parser = argparse.ArgumentParser(
        description='Run predictions with a trained model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
