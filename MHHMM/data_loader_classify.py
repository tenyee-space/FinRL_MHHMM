import numpy
from torch.utils.data import Dataset, DataLoader
import csv
import torch
import argparse
import pandas as pd


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
        self.load_new()
        # self.load()

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

    def normal_data_new(self):
        # 将数据转换为 DataFrame 以便处理
        # data = pd.DataFrame(self.features, columns=[
        #     'open', 'high', 'low', 'close', 'volume', 'day', 'macd', 'boll_ub',
        #     'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma',
        #     'vix', 'turbulence'
        # ])
        data = pd.DataFrame(self.features, columns=[
            'Open', 'High', 'Low', 'Close', 'Volume'
        ])
        
        # 遍历每一列进行归一化处理
        for column in data.columns:
            # 计算非异常值的均值
            non_abnormal = data[column][self.record_abnormal[:, data.columns.get_loc(column)] == 0]
            mean = non_abnormal.mean()
            
            # 填补异常值
            data.loc[self.record_abnormal[:, data.columns.get_loc(column)] == 1, column] = mean
            
            # 计算方差
            var = non_abnormal.var()
            
            # 标准化处理
            data[column] = (data[column] - mean) / numpy.sqrt(var + 1e-9)

        # 手动合并特征
        data['price_range'] = (data['high'] - data['low'])  #  价差
        data['moving_avg_diff'] = (data['close_30_sma'] - data['close_60_sma'])  # 短期移动平均线与长期移动平均线的差值

        # 保留3-4个新特征
        selected_features = data[['price_range', 'moving_avg_diff', 'volume']].values

        # 更新 self.features
        self.features = selected_features
    
    def print_features(self):
        print(self.features)

    def load_new(self):
        # 读取CSV文件
        data = pd.read_csv(self.files_path)
        # 截取前8000行
        data = data.iloc[:4000, :]
        
        # 提取特征列
        # feature_columns = [
        #     'open', 'high', 'low', 'close', 'volume', 'day', 'macd', 'boll_ub',
        #     'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma',
        #     'vix', 'turbulence'
        # ]
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]
        self.features = data[feature_columns].values
        
        # 提取标签列
        labels_data = pd.read_csv(self.labels_path)
        labels_data = labels_data.iloc[:4000, :]
        self.labels = numpy.zeros_like(labels_data['Close'].values)
        
        # 更新最大值和最小值
        self.max = numpy.max(self.features)
        self.min = numpy.min(self.features)
        
        # 将特征和标签转换为numpy数组
        self.features = numpy.array(self.features, dtype=float)
        self.labels = numpy.array(self.labels, dtype=float)

        # 记录异常值
        self.record_abnormal = numpy.isnan(self.features).astype(int)
        
        # 填充异常值
        self.features = numpy.nan_to_num(self.features)

        # 计算数据长度
        self.length = len(self.labels)
        
        # 确保 self.record_abnormal 是一个 NumPy 数组
        self.record_abnormal = numpy.array(self.record_abnormal)

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
                file_path = self.files_path[0:-5] + str(i + 1) + self.files_path[-5:]
            else:
                file_path = self.files_path[0:-5] + str(i + 1) + self.files_path[-5:]
            data = pd.read_excel(file_path)
            for index in range(size):  # 根据train-size确定行数
                for t in range(self.dataset_args.max_series_length):  # 根据series-length确定列数
                    colcell = data.iloc[index, t]
                    if pd.isnull(colcell):
                        self.record_abnormal[index][t][i] = 1
                    else:
                        self.features[index][t][i] = float(colcell)
                        self.max = max(self.max, float(colcell))
                        self.min = min(self.min, float(colcell))

        label = pd.read_excel(self.labels_path)
        for index in range(len(label)):
            row = label.iloc[index]
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
    parser_dataset.add_argument('--max-series-length', type=int, default=50, help='None')
    parser_dataset.add_argument('--num-class', type=int, default=2, help='None')
    args_dataset = parser_dataset.parse_args()
    train_dataset = load_dataset(args_dataset, '../FingerMovements/train_dim.xlsx',
                                 '../FingerMovements/train_label.xlsx', train=True)
    test_dataset = load_dataset(args_dataset, '../FingerMovements/test_dim.xlsx',
                                '../FingerMovements/test_label.xlsx', train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=False)
    for ibatch, data in enumerate(train_loader):
        features = data[0]
        y = data[1]
        print(features)
        print(y)
        print(features.shape)
        print(y.shape)
