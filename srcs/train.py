from model import LinearRegression
from dataset import Dataset

import argparse

def save(model, output_file):
    print (f'Saving to {output_file}:')

def plot(model):
    pass

def loss(model, features, output):
    error = (model.predict(features) - output) ** 2
    return error

def train(args):
    model = LinearRegression()
    dataset = Dataset(args.input_file)
    print ('Training...')
    for _ in range(100):
        theta0 = 0
        theta1 = 0
        for features, output in zip(dataset.x, dataset.y):
            pass
            # error = loss(model, features, output)
            # theta0 += error
            # theta1 += error * features[0]
        theta0 *= 2 / len(dataset.x)
        theta1 *= 2 / len(dataset.y)
        model.bias += -0.01 * theta0
        model.weights[0] += -0.01 * theta1
    return model

def get_arguments():
    parser = argparse.ArgumentParser(
        description = 'Train a linear regression model for car price prediction.'
    )
    parser.add_argument(
        'input_file',
        help = 'CSV file containing car features (including the price)'
    )
    parser.add_argument(
        '-output_file',
        default='parameters',
        help = 'Output file that will contain the trained parameters of the model'
    )
    return parser.parse_args()

def main():
    arguments = get_arguments()
    model = train(arguments)
    save(model, arguments.output_file)

if __name__ == '__main__':
    main()
