import numpy
from torch.utils.data import Dataset, DataLoader
import csv
import torch
import argparse


class load_dataset(Dataset):
    def __init__(self, dataset_args, files_path, labels_path, train=True):
        self.dataset_args = dataset_args
        self.files_path = files_path
        self.labels_path = labels_path
        self.train = train
        self.features = []
        self.labels = []
        self.length = []
        self.record_abnormal = []
        self.word_tables = dict()
        self.labels_count = 0
        self.max = -numpy.inf
        self.min = numpy.inf
        self.load()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.float)

        return features, y

    def get_min_max(self):
        return self.min, self.max

    def normal_data(self):
        if self.train:
            size = self.dataset_args.train_size
        else:
            size = self.dataset_args.test_size
        for i in range(size):
            count = 0
            sum = 0
            for j in range(self.dataset_args.max_series_length):
                for k in range(self.dataset_args.num_dim):
                    if self.record_abnormal[i][j][k] == 0:
                        count += 1
                        sum += self.features[i, j, k]
            mean = sum / count
            sum = 0
            for j in range(self.dataset_args.max_series_length):
                for k in range(self.dataset_args.num_dim):
                    if self.record_abnormal[i][j][k] == 1:
                        self.features[i, j, k] = mean
                    sum += (self.features[i, j, k] - mean) ** 2
            var = sum / count
            self.features[i] = (self.features[i] - mean) / numpy.sqrt(var + 1e-9)

    def print_features(self):
        print(self.features)

    def load(self):
        if self.train:
            size = self.dataset_args.train_size
        else:
            size = self.dataset_args.test_size
        for i in range(size):
            self.features.append([])
            self.record_abnormal.append([])
            for j in range(self.dataset_args.max_series_length):
                self.features[i].append([])
                self.record_abnormal[i].append([])
                for k in range(self.dataset_args.num_dim):
                    self.features[i][j].append(0)
                    self.record_abnormal[i][j].append(0)

        for i in range(self.dataset_args.num_dim):
            if self.train:
                file_path = self.files_path[0:-11] + str(i + 1) + self.files_path[-10:]
            else:
                file_path = self.files_path[0:-10] + str(i + 1) + self.files_path[-9:]
            with open(file_path, 'r') as f:
                rdr = csv.reader(f)
                for index, row in enumerate(rdr):
                    for t, colcell in enumerate(row):
                        if colcell == '?':
                            self.record_abnormal[index][t][i] = 1
                        else:
                            self.features[index][t][i] = float(colcell)
                            self.max = max(self.max, float(colcell))
                            self.min = min(self.min, float(colcell))

        with open(self.labels_path, 'r') as f:
            rdr = csv.reader(f)
            for index, row in enumerate(rdr):
                if row[0] in self.word_tables.keys():
                    self.labels.append(self.word_tables[row[0]])
                else:
                    self.word_tables[row[0]] = self.labels_count
                    self.labels.append(self.word_tables[row[0]])
                    self.labels_count += 1

        self.features = numpy.array(self.features, dtype=float)
        self.labels = numpy.array(self.labels, dtype=float)


if __name__ == '__main__':
    parser_dataset = argparse.ArgumentParser(description='dataset')
    parser_dataset.add_argument('--train-size', type=int, default=316, help='None')
    parser_dataset.add_argument('--test-size', type=int, default=100, help='None')
    parser_dataset.add_argument('--num-dim', type=int, default=28, help='None')
    parser_dataset.add_argument('--series-length', type=int, default=50, help='None')
    parser_dataset.add_argument('--num-class', type=int, default=2, help='None')
    args_dataset = parser_dataset.parse_args()
    train_dataset = load_dataset(args_dataset, './data/FingerMovementsDimensionX_TRAIN.csv',
                                 './data/train_label.csv', train=True)
    test_dataset = load_dataset(args_dataset, './data/FingerMovementsDimensionX_TEST.csv',
                                './data/test_label.csv', train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=False)
    for ibatch, data in enumerate(train_loader):
        features = data[0]
        y = data[1]
        print(features)
        print(y)
        print(features.shape)
        print(y.shape)
