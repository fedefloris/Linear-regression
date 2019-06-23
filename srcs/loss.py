def mse(model, dataset):
    error = (model.predict(dataset.x) - dataset.y) ** 2
    return error.sum() / len(dataset.x)

def mae(model, dataset):
    error = abs(model.predict(dataset.x) - dataset.y)
    return error.sum() / len(dataset.x)

def mape(model, dataset):
    error = abs(model.predict(dataset.x) - dataset.y) / dataset.y
    return error.sum() * (100 / len(dataset.x))

def mpe(model, dataset):
    error = (model.predict(dataset.x) - dataset.y) / dataset.y
    return error.sum() * (100 / len(dataset.x))