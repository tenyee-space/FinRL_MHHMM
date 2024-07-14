from __future__ import print_function
from torch import optim
from model import Model
import torch
from tensorboardX import SummaryWriter
import os
from datetime import datetime
from train_classify import train
from val_classify import val
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
     def load_pretrained(dict1, dict2):
        """
        加载预训练权重，键（网络层名）或值（网络权重）的形状不一致时保留dict2的键值
        :param dict1: 预训练网络的权重
        :param dict2: 目标网络的权重
        :return: dict2
        """
        for k, v in dict2.items():
            if k in dict1 and v.shape == dict1[k].shape:
                dict2[k] = dict1[k]
                print(f'load {k} layer')
            else:
                print(f'cannot load {k} layer')
        return dict2

     def val(self, dataset, batch_size, writer, epoch, n_iter, log):
        return val(self.args, self.model, dataset, batch_size, writer, epoch, n_iter, log)

     def kl_loss_function(self, recon_input, input, mu, logvar, mu_prior, logvar_prior, args):
        """
        计算重构损失和KL散度。
        :param recon_input: 由模型生成的重构输入。
        :param input: 原始输入数据。
        :param mu: 编码后的均值。
        :param logvar: 编码后的对数方差。
        :param mu_prior: 先验均值。
        :param logvar_prior: 先验对数方差。
        :param args: 包含模型参数的对象。
        :return: 重构损失和KL散度。
        """
        BCE = torch.mean((recon_input - input) ** 2)
        KLD = 0.5 * torch.mean(torch.mean(logvar.exp() / logvar_prior.exp() +
                                          (mu - mu_prior).pow(2) / logvar_prior.exp() +
                                          logvar_prior - logvar - 1, dim=1))
        return BCE * args.vae_weight, KLD * args.kl_weight

     def validate(self, val_dataset, batch_size, writer, epoch, n_iter, val_log):
        """
        执行模型验证。
        :param val_dataset: 验证数据集。
        :param batch_size: 批处理大小。
        :param writer: 用于记录TensorBoard日志的writer对象。
        :param epoch: 训练周期。
        :param n_iter: 迭代次数。
        :param val_log: 用于记录验证日志的logger对象。
        :return: 累计的重构损失、KL散度、预测损失、准确率和F1分数。
        """
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, drop_last=False, shuffle=False)
        self.model.eval() 
        rec_loss_input, kl_loss_input, pre_loss = 0, 0, 0
        all_gt_y, all_pre_id = [], []
        with torch.no_grad(): 
            for i_batch, sample_batched in enumerate(val_loader):
                inputs = torch.tensor(sample_batched[0], device=self.args.device)
                gt_y = torch.tensor(sample_batched[1], device=self.args.device)

                # 重置隐藏状态
                batch_size = gt_y.shape[0]
                h0 = torch.zeros(batch_size, self.args.h_dim, device=self.args.device)
                self.model.reset(h0)
                h_t_last = h0

                all_rec_loss, all_kl_loss = torch.zeros(1, device=self.args.device), torch.zeros(1, device=self.args.device)
                logit = None

                # 处理每个时间步
                for index in range(len(inputs[0])):
                    if index == 0:
                        input_t = inputs[:, index]
                        input_t_last = torch.zeros_like(input_t)
                        h_t_last = h0
                    else:
                        input_t = inputs[:, index]
                        input_t_last = inputs[:, index - 1]

                    if input_t.sum() == 0:
                        break

                    # 模型前向传播
                    vae_rec_input_t, mu_h, logvar_h, mu_h_prior, logvar_h_prior = self.model(input_t, test=True)
                    loss_re_x_t, loss_kl_x_t = self.kl_loss_function(vae_rec_input_t, input_t, mu_h, logvar_h, mu_h_prior, logvar_h_prior, self.args)
                    all_rec_loss += loss_re_x_t
                    all_kl_loss += loss_kl_x_t

                    if logit is None:
                        logit = self.model.classify_one_step()
                    else:
                        logit += self.model.classify_one_step()

                mean_rec_loss = all_rec_loss / len(inputs)
                mean_kl_loss = all_kl_loss / len(inputs)

                rec_loss_input += mean_rec_loss.item()
                kl_loss_input += mean_kl_loss.item()

                # 应用Softmax进行分类概率计算
                func = nn.Softmax(dim=1)
                prob = func(logit)
                prob0 = prob[:, 0].cpu().detach().numpy()
                prob1 = prob[:, 1].cpu().detach().numpy()

                all_gt_y.extend(gt_y.cpu().numpy())
                all_pre_id.extend(torch.max(logit, 1)[1].data.cpu().numpy())

                # 计算分类损失
                cn_loss = nn.CrossEntropyLoss()
                loss_pre = cn_loss(logit, gt_y.long())
                pre_loss += loss_pre.item()

                # 记录损失和性能指标
                writer.add_scalar('val/mean rec loss of image', loss_re_x, n_iter)
                writer.add_scalar('val/mean kl loss of image', loss_kl_x, n_iter)
                writer.add_scalar('val/classification loss', loss_pre, n_iter)

                n_iter += 1

        # 计算总体性能指标
        acc = accuracy_score(all_gt_y, all_pre_id)
        f1 = f1_score(all_gt_y, all_pre_id, average='micro')
        all_val_acc = (all_pre_id == all_gt_y).sum() / len(all_gt_y)

        # 记录最终结果
        writer.add_scalar('val/AUC', acc, epoch)
        writer.add_scalar('val/f1', f1, epoch)
        writer.add_scalar('val/ACC', all_val_acc, epoch)
        val_log.info(f"Validation - Epoch {epoch}: Rec Loss {rec_loss_input}, KL Loss {kl_loss_input}, Pre Loss {pre_loss}, ACC {all_val_acc}, F1 {f1}")

        return rec_loss_input, kl_loss_input, pre_loss, all_val_acc, f1

     def train(self, train_dataset, batch_size, optimizer_hmm, writer, epoch, n_iter, train_log):
        return train(self.args, self.model, train_dataset, batch_size, optimizer_hmm,writer, epoch, n_iter, train_log)

if __name__ == '__main__':
    args = ...  # 需要提供参数
    hmm_model = DCHMM(args)
    # 假设已经有了验证数据集等其他参数
    accuracy = hmm_model.val(val_dataset, batch_size, writer, epoch, n_iter, val_log)
    print(f'Validation accuracy: {accuracy}')