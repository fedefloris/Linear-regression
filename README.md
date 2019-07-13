# Linear-regression - 42born2code
[![Build Status](https://travis-ci.com/fedefloris/Linear-regression.svg?token=dH8C3CpkpNBzxeKzZ8gb&branch=master)](https://travis-ci.com/fedefloris/Linear-regression)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

<p align="center">
	<img width="750" src="https://github.com/fedefloris/Linear-regression/blob/master/images/preview.png">
</p>

## Challenge
A multiple [linear regression](https://en.wikipedia.org/wiki/Linear_regression) model trained with [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm.

The code supports multiple independent variables (but not categorical, only numbers).
It can read a dataset with the following format:

<b>N</b> = number of training examples

| Column 1 | Column 2 | Column 3 |
| :--: | :--: | :--: |
| x<sub>1,1</sub> | x<sub>1,2</sub> |  y<sub>1</sub> |
| x<sub>2,1</sub> | x<sub>2,2</sub> |  y<sub>2</sub> |
| x<sub>3,1</sub> | x<sub>3,2</sub> |  y<sub>3</sub> |
| ... | ... | ... |
| x<sub><b>N</b>,1</sub> | x<sub><b>N</b>,2</sub> |  y<sub><b>N</b></sub> |

Example with [data/cars.csv](data/cars.csv), a tiny dataset that contains kilometers-prices pairs.
```console
$> head -n 4 data/cars.csv
km,price
240000,3650
139800,3800
150500,4400
```

For more details look at the [subject](subject.pdf).

## Install dependencies
```console
pip install -r requirements.txt
```

## Using the project
#### Train the model and plot the results
```console
python srcs/train.py data/cars.csv -plot
```
#### Evaluate the trained model
```console
python srcs/evaluate.py data/cars.csv
```
#### Predict
```console
python srcs/predict.py
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
