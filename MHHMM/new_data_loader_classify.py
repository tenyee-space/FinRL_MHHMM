import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 这个类用于实现加载csv数据集
# csv格式为：
# Data columns (total 19 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   Unnamed: 0    83897 non-null  int64  
#  1   date          83897 non-null  object 
#  2   tic           83897 non-null  object 
#  3   open          83897 non-null  float64
#  4   high          83897 non-null  float64
#  5   low           83897 non-null  float64
#  6   close         83897 non-null  float64
#  7   volume        83897 non-null  float64
#  8   day           83897 non-null  float64
#  9   macd          83897 non-null  float64
#  10  boll_ub       83897 non-null  float64
#  11  boll_lb       83897 non-null  float64
#  12  rsi_30        83897 non-null  float64
#  13  cci_30        83897 non-null  float64
#  14  dx_30         83897 non-null  float64
#  15  close_30_sma  83897 non-null  float64
#  16  close_60_sma  83897 non-null  float64
#  17  vix           83897 non-null  float64
#  18  turbulence    83897 non-null  float64

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, 3:].values  # 从第4列开始是特征
        self.labels = self.data['tic'].astype('category').cat.codes.values  # 将'tic'列转换为类别编码

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return sample

def load_custom_dataset(csv_file, batch_size, shuffle=True):
    dataset = CustomDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader