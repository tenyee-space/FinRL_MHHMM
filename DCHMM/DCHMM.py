from __future__ import print_function
from torch import optim
from model import Model
import torch
from tensorboardX import SummaryWriter
import os
from datetime import datetime
from train_classify import train
from val_classify import val,val_with_gt_pre
from data_loader_classify import load_dataset
import logging
import warnings
import argparse

class DCHMM:
     def __init__(self, args):
        self.args = args
        self.model = self.initialize_model()
        self.device = args.device
        self.model.to(self.device)
        if args.load_pre_train_weight:
            self.load_pretrained_weights()

     def initialize_model(self):
        """
        初始化DCHMM模型。
        """
        return Model(self.args)

     def load_pretrained_weights(self):
        print('Loading pre-train weight')
        print(f'The pre-train weight\'s path is {self.args.pre_train_weight}')
        try:
            pre_train_weight = torch.load(self.args.pre_train_weight, map_location=self.device)
            self.model.load_state_dict(load_pretrained(pre_train_weight, self.model.state_dict()))
            del pre_train_weight
        except FileNotFoundError:
            print('Pre-train model not found, loading nothing.')

     def val(self, dataset, batch_size, writer, epoch, n_iter, log):
        return val(self.args, self.model, dataset, batch_size, writer, epoch, n_iter, log)

     def val_with_gt_pre(self, dataset, batch_size, writer, epoch, n_iter, log):
        return val_with_gt_pre(self.args, self.model, dataset, batch_size, writer, epoch, n_iter, log)

     def train(self, train_dataset, batch_size, optimizer_hmm, writer, epoch, n_iter, train_log):
        return train(self.args, self.model, train_dataset, batch_size, optimizer_hmm,writer, epoch, n_iter, train_log)

if __name__ == '__main__':
    args = ...  # 需要提供参数
    hmm_model = DCHMM(args)
    # 假设已经有了验证数据集等其他参数
    accuracy = hmm_model.val(val_dataset, batch_size, writer, epoch, n_iter, val_log)
    print(f'Validation accuracy: {accuracy}')