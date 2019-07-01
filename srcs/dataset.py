import numpy as np
import csv

class Dataset:

    def __init__(self,
            file,
            delimiter=',',
            preprocess=True
        ):
        self._read_data_from_csv(file, delimiter)
        if preprocess:
            self.x = Dataset.preprocess(self.x)

    def _read_data_from_csv(self, csv_file, delimiter):
        self.x, self.y = [], []
        with open(csv_file) as file:
            reader = csv.reader(file, delimiter=delimiter)
            # Skip first row of columns
            next(reader, None)
            for row in reader:
                self.x.append(list(map(float, row[:-1])))
                self.y.append([float(row[-1])])
            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.x_raw = self.x

    @staticmethod
    def preprocess(inputs):
        # min-max scaling
        inputs_min = np.amin(inputs, 0)
        inputs_max = np.amax(inputs, 0)
        return (inputs - inputs_min) * (1 / (inputs_min - inputs_max))