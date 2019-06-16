import csv

class Dataset:

    def __init__(self,
            file,
            delimiter = ','
        ):
        self._read_data_from_csv(file, delimiter)

    def _read_data_from_csv(self, csv_file, delimiter):
        self.x = []
        self.y = []
        with open(csv_file) as file:
            reader = csv.reader(file, delimiter=delimiter)
            # Skip first row columns
            next(reader)
            for row in reader:
                self.x.append(list(map(float, row[:-1])))
                self.y.append(float(row[-1]))