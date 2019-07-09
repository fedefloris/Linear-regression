# Linear-regression - 42born2code
[![Build Status](https://travis-ci.com/fedefloris/Linear-regression.svg?token=dH8C3CpkpNBzxeKzZ8gb&branch=master)](https://travis-ci.com/fedefloris/Linear-regression)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

<p align="center">
	<img width="750" src="https://github.com/fedefloris/Linear-regression/blob/master/images/preview.png">
</p>

## Challenge
A simple [linear regression](https://en.wikipedia.org/wiki/Linear_regression) model.

By default, the program reads [data/cars.csv](data/cars.csv), a tiny dataset that contains kilometers-prices pairs.

Here, you can see the first 4 lines:
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
python srcs/train.py -plot
```
#### Evaluate the trained model
```console
python srcs/evaluate.py
```
#### Predict
```console
python srcs/predict.py
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
