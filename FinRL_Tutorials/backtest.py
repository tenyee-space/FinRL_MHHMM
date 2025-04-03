import pandas as pd
import numpy as np
import os
import torch
import logging

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR

from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

from MHHMM.MHHMM import DCHMM

train = pd.read_csv('train_data.csv')
trade = pd.read_csv('trade_data.csv')

# If you are not using the data generated from part 1 of this tutorial, make sure
# it has the columns and index in the form that could be make into the environment.
# Then you can comment and skip the following lines.
train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

# 创建自定义Dataset类
class StockTradingDataset(Dataset):
    def __init__(self, env, volume_threshold=0.05):
        """
        将StockTradingEnv环境中的数据转换为DCHMM可用的数据集
        
        Args:
            env: StockTradingEnv实例
            volume_threshold: 交易量波动阈值，默认为5%
        """
        self.df = env.df  # 获取环境中的DataFrame
        self.tech_indicator_list = env.tech_indicator_list
        self.stock_dim = env.stock_dim
        self.volume_threshold = volume_threshold
        
        # 提取特征和标签
        self.features = []
        self.labels = []
        self.record_abnormal = []
        
        # 处理每个时间步的数据
        dates = self.df.index.unique()
        for i in range(len(dates) - 1):  # 减1是因为需要下一天的数据来生成标签
            date = dates[i]
            next_date = dates[i+1]
            
            # 获取当前日期的所有股票数据
            current_data = self.df[self.df.index == date]
            next_data = self.df[self.df.index == next_date]
            
            # 提取特征（使用OHLCV和技术指标）
            feature_list = []
            for stock_idx in range(self.stock_dim):
                stock_tic = current_data.tic.unique()[stock_idx]
                stock_data = current_data[current_data.tic == stock_tic]
                
                # 基本价格数据
                stock_features = [
                    stock_data['open'].values[0],
                    stock_data['high'].values[0],
                    stock_data['low'].values[0],
                    stock_data['close'].values[0],
                    stock_data['volume'].values[0]
                ]
                
                # 技术指标
                for indicator in self.tech_indicator_list:
                    if indicator in stock_data.columns:
                        stock_features.append(stock_data[indicator].values[0])
                
                feature_list.extend(stock_features)
            
            self.features.append(feature_list)
            
            # 标签生成逻辑：基于交易量波动
            # 计算当前和下一天的交易量
            current_volumes = current_data['volume'].values
            next_volumes = next_data['volume'].values
            
            # 计算交易量变化百分比
            volume_changes = []
            for j in range(len(current_volumes)):
                if j < len(next_volumes):
                    if current_volumes[j] > 0:
                        change = abs(next_volumes[j] - current_volumes[j]) / current_volumes[j]
                        volume_changes.append(change)
            
            # 计算平均交易量变化
            if volume_changes:
                avg_volume_change = np.mean(volume_changes)
                # 如果交易量波动小于阈值，标签为1，否则为0
                label = 1 if avg_volume_change < self.volume_threshold else 0
            else:
                label = 0  # 默认标签
            
            self.labels.append(label)
            
            # 记录异常值（NaN）
            abnormal = np.isnan(feature_list).astype(int)
            self.record_abnormal.append(abnormal)
        
        # 转换为numpy数组
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.record_abnormal = np.array(self.record_abnormal, dtype=np.int32)
        
        # 处理NaN值
        self.features = np.nan_to_num(self.features)
        
        # 计算最大值和最小值
        self.max = np.max(self.features)
        self.min = np.min(self.features)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return features, y
    
    def get_min_max(self):
        return self.min, self.max
    
    def normal_data_new(self):
        """标准化数据"""
        for col in range(self.features.shape[1]):
            # 计算非异常值的均值和方差
            non_abnormal = self.features[:, col][self.record_abnormal[:, col] == 0]
            if len(non_abnormal) > 0:
                mean = np.mean(non_abnormal)
                var = np.var(non_abnormal)
                
                # 填补异常值
                self.features[:, col][self.record_abnormal[:, col] == 1] = mean
                
                # 标准化
                self.features[:, col] = (self.features[:, col] - mean) / np.sqrt(var + 1e-9)
    
    def print_features(self):
        """打印特征样本"""
        print("特征样本前5条:")
        print(self.features[:5])
        print("标签样本前5条:")
        print(self.labels[:5])

# 设置日志
def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

# 创建日志目录
log_dir = '../FinRL_DCHMM/logs'
os.makedirs(log_dir, exist_ok=True)

# 设置日志
val_log = {}
setup_logger('val_log', os.path.join(log_dir, 'val_logger'))
val_log['val_log'] = logging.getLogger('val_log')

# 创建DCHMM模型所需的参数
class Args:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_pre_train_weight = False
        self.vae_weight = 1.0
        self.kl_weight = 10.0
        self.h_dim = 3  # 隐藏层维度
        self.input_dim = 5 + len(INDICATORS)  # 特征维度：OHLCV + 技术指标
        self.hidden_dim = 3
        self.k = 3
        self.m = 3
        self.class_num = 2  # 二分类问题
        self.input_fc_dim = 3
        self.cls_fc_dim = 3
        self.batch_size = 1

# 创建DCHMM模型
args = Args()
dchmm_model = DCHMM(args)

# 创建数据集
val_dataset = StockTradingDataset(e_trade_gym, volume_threshold=0.05)
val_dataset.normal_data_new()
val_dataset.print_features()

# 设置验证参数
batch_size = 4
writer = SummaryWriter(os.path.join(log_dir, 'Train'))
epoch = 0
n_iter = 0

# 调用val_with_gt_pre方法
all_gt_y, all_pre_id = dchmm_model.val_with_gt_pre(val_dataset, batch_size, writer, epoch, n_iter, val_log)

# 并行用强化学习来执行预测
df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c,
    environment = e_trade_gym) if if_using_a2c else (None, None)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg,
    environment = e_trade_gym) if if_using_ddpg else (None, None)

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo,
    environment = e_trade_gym) if if_using_ppo else (None, None)

df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_td3,
    environment = e_trade_gym) if if_using_td3 else (None, None)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac,
    environment = e_trade_gym) if if_using_sac else (None, None)