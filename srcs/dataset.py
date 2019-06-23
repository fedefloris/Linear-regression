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
            self._preprocess()

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

    def _preprocess(self):
        # min-max scaling
        x_min = np.amin(self.x, 0)
        x_max = np.amax(self.x, 0)
        self.x = (self.x - x_min) * (1 / (x_min - x_max))