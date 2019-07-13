from dataset import Dataset
from model import LinearRegression
from loss import mse, mae, mpe, mape

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import argparse

def display_sklearn_metrics(dataset):
    model = SklearnLinearRegression().fit(dataset.x, dataset.y)
    print ('--- Sklearn model\'s metrics ---')
    print ('Weights =', model.coef_.flatten(), 'Bias =', model.intercept_[0])
    print ('Mean square error:', mean_squared_error(dataset.y, model.predict(dataset.x)))
    print ('Mean absolute error:', mean_absolute_error(dataset.y, model.predict(dataset.x)))


def display_model_metrics(model, dataset):
    print ('--- Model\'s metrics ---')
    print (model)
    print ('Mean square error:', mse(model, dataset))
    print ('Mean absolute error:', mae(model, dataset))
    print ('Mean percentage error: ', mpe(model, dataset))
    print ('Mean absolute percentage error: ', mape(model, dataset), '\n')

def evaluate(args):
    dataset = Dataset(args.dataset)
    model = LinearRegression()
    print ('Loading parameters from', args.model_parameters, '\n')
    model.load_parameters(args.model_parameters)
    display_model_metrics(model, dataset)
    display_sklearn_metrics(dataset)

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a linear regression model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset',
        help='Dataset with each sample separated by a new line and each variable separated by a comma'
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
