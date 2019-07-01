import numpy as np
import csv

class Dataset:

    def __init__(self,
            file,
            delimiter=',',
            preprocess=True
        ):
        self._read_data_from_csv(file, delimiter)
        self._configure()
        if preprocess:
            self.x = Dataset.preprocess(
                self.x,
                self.x_max,
                self.x_min
            )

    def _read_data_from_csv(self, csv_file, delimiter):
        self.x, self.y = [], []
        with open(csv_file) as file:
            reader = csv.reader(file, delimiter=delimiter)
            # Skip first row of columns
            next(reader, None)
            for row in reader:
                self.x.append(list(map(float, row[:-1])))
                self.y.append([float(row[-1])])

    def _configure(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x_raw = self.x
        self.x_max = np.amax(self.x, 0)
        self.x_min = np.amin(self.x, 0)

    @staticmethod
    # min-max scaling
    def preprocess(inputs, inputs_max, inputs_min):
        return (inputs - inputs_min) / (inputs_max - inputs_min)