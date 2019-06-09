import csv

class Dataset:

    def __init__(self,
            file,
            delimiter = ','
        ):
        self._read_data_from_csv(file, delimiter)

    def _read_data_from_csv(self, csv_file, delimiter):
        self.x, self.y = [], []
        with open(csv_file) as file:
            reader = csv.reader(file, delimiter=delimiter)
            x, y = [], []
            for row in reader:
                x.append(row[0])
                y.append(row[1])
            self.x = x[1:]
            self.y = y[1:]