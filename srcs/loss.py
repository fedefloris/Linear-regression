def mse(model, dataset):
    errors = (model.predict(dataset.x) - dataset.y) ** 2
    return errors.sum() / len(dataset.y)

def mae(model, dataset):
    errors = abs(model.predict(dataset.x) - dataset.y)
    return errors.sum() / len(dataset.y)

def mape(model, dataset):
    errors = abs(model.predict(dataset.x) - dataset.y) / dataset.y
    return errors.sum() * (100 / len(dataset.y))

def mpe(model, dataset):
    errors = (model.predict(dataset.x) - dataset.y) / dataset.y
    return errors.sum() * (100 / len(dataset.y))