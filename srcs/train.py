from model import LinearRegression

import argparse

def save(model):
    print ('Saving to ...:')

def plot(model):
    pass

def train(model):
    print ('Training...')
    print (model)

def main():
    model = LinearRegression()
    train(model)
    save(model)

if __name__ == '__main__':
    main()