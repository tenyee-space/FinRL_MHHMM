from __future__ import print_function
from torch import optim
from model import *
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
from DCHMM import *

warnings.filterwarnings('ignore')

dataset_list = {
   'SPI': {'name': 'SharePriceIncrease', 'train_size': '8000', 'test_size': '9715', 'dims': '2', 'length': '60',
            'classes': '2', 'batch_size': '4'}}

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


parser1 = argparse.ArgumentParser(description='pre-input')
parser1.add_argument('--target-dataset', type=str, default='SPI', help='None')
parser1.add_argument('--machine-idx', type=str, default='v1', help='None')
parser1.add_argument('--v-dims', type=int, default=3, help='None')
args1 = parser1.parse_args()


target_dataset = args1.target_dataset
machine_idx = args1.machine_idx
v_dims = args1.v_dims

parser = argparse.ArgumentParser(description='our')
parser.add_argument('--datasetname', type=str, default=dataset_list[target_dataset]['name'], help='None')
parser.add_argument('--dataset-dir', type=str, default=r'/root/autodl-tmp', help='None')
parser.add_argument('--train-size', type=int, default=dataset_list[target_dataset]['train_size'], help='None')
parser.add_argument('--test-size', type=int, default=dataset_list[target_dataset]['test_size'], help='None')
parser.add_argument('--num-dim', type=int, default=dataset_list[target_dataset]['dims'], help='None')
parser.add_argument('--max-series-length', type=int, default=dataset_list[target_dataset]['length'], help='None')
parser.add_argument('--num-class', type=int, default=dataset_list[target_dataset]['classes'], help='None')
parser.add_argument('--series-length-file', type=str, default='', help='the-length-file-path')

parser.add_argument('--input_dim', type=int, default=dataset_list[target_dataset]['dims'], help='None')
parser.add_argument('--hidden_dim', type=int, default=v_dims, help='None')
# parser.add_argument('--k', type=int, default=5, help='None')
parser.add_argument('--k', type=int, default=3, help='None')
parser.add_argument('--m', type=int, default=3, help='None')
parser.add_argument('--class_num', type=int, default=dataset_list[target_dataset]['classes'], help='None')
parser.add_argument('--input_fc_dim', type=int, default=v_dims, help='input_fc_dim')
parser.add_argument('--h-dim', type=int, default=v_dims, metavar='h', help='latent size of h in vae (default: 128)')

parser.add_argument('--device', type=str, default='cuda:0', help='None')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.0001)')
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--batch-size', type=int, default=dataset_list[target_dataset]['batch_size'], help='batch size')
parser.add_argument('--init', default='xavier', help='default, this can not be optional')
parser.add_argument('--optimizer', default='Adam', help='optimizer (Adam or RMSprop, SGD, Adagrad, Momentum, Adadelta)')
parser.add_argument('--kl-weight', type=float, default=10, metavar='LR', help='KL loss weight (default: 0.1)')
parser.add_argument('--vae-weight', type=float, default=1, metavar='LR', help='VAE loss weight (default: 0.1)')
parser.add_argument('--cls-loss-weight', type=float, default=10, metavar='LR',
                    help='classification loss weight (default: 0.1)')
parser.add_argument('--cls-fc-dim', type=int, default=v_dims, help='cls FC dim')
parser.add_argument('--load-pre-train-weight', type=bool, default=False,
                    help='whether or not need to load the pre-train weight')
parser.add_argument('--pre-train-weight', type=str,
                    default='./classify_'+ machine_idx +'/HMM_model_' + target_dataset + '_pretrain.pkl',
                    help='VAE-pre-train-weight-path')
parser.add_argument('--random-seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
args, unknown = parser.parse_known_args()
print(vars(args))
device = args.device

def main():
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    device = args.device

    hmm_model = DCHMM(args)

    lr = args.lr

    optimizer_hmm = optim.Adam(hmm_model.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler_hmm = torch.optim.lr_scheduler.MultiStepLR(optimizer_hmm, milestones=[10, 30, 50, 80], gamma=0.9)
    current_time = datetime.now().strftime('%b%d_%H-%M')
    print('current_time', current_time)
    save_dir = os.path.join('./classify' + '_' + machine_idx, current_time + '_' + target_dataset)
    writer = SummaryWriter(os.path.join(save_dir, 'Train'))

    # save params log
    log = {}
    setup_logger('params_log', r'{0}/logger'.format(save_dir))
    log['params_log'] = logging.getLogger('params_log')
    d_args = vars(args)
    for k in d_args.keys():
        log['params_log'].info('{0}: {1}'.format(k, d_args[k]))

    dataset_log = {}
    setup_logger('dataset_log', r'{0}/dataset_logger'.format(save_dir))
    dataset_log['dataset_log'] = logging.getLogger('dataset_log')
    d_args = vars(args)
    for k in d_args.keys():
        dataset_log['dataset_log'].info('{0}: {1}'.format(k, d_args[k]))

    train_log = {}
    setup_logger('train_log', r'{0}/train_logger'.format(save_dir))
    train_log['train_log'] = logging.getLogger('train_log')

    val_log = {}
    setup_logger('val_log', r'{0}/val_logger'.format(save_dir))
    val_log['val_log'] = logging.getLogger('val_log')

    train_epoch = args.epoch
    batch_size = args.batch_size

    train_set_path = '/root/FinRL_DCHMM/DCHMM/20240917data.csv'
    train_label_path = '/root/FinRL_DCHMM/DCHMM/20240917data.csv'
    val_set_path = '/root/FinRL_DCHMM/DCHMM/20240917data.csv'
    val_label_path = '/root/FinRL_DCHMM/DCHMM/20240917data.csv'

    train_dataset = load_dataset(args, train_set_path, train_label_path, train=True)
    val_dataset = load_dataset(args, val_set_path, val_label_path, train=False)

    mi1, ma1 = train_dataset.get_min_max()
    mi2, ma2 = val_dataset.get_min_max()
    mi = min(mi1, mi2)
    ma = max(ma1, ma2)
    print(mi, ma)
    del mi1, mi2, mi, ma1, ma2, ma

    train_dataset.normal_data_new()
    val_dataset.normal_data_new()

    train_dataset.print_features()
    val_dataset.print_features()

    print('train num', train_dataset.__len__(), 'val num',
          val_dataset.__len__())

    n_iter = 0
    current_best_acc = 0
    last_epoch = 0

    for epoch in range(train_epoch):
        print('training in epoch {0}'.format(epoch))
        n_iter = hmm_model.train(train_dataset, batch_size, optimizer_hmm, writer, epoch, n_iter, train_log)

        scheduler_hmm.step()

        acc_val_value = hmm_model.val(val_dataset, batch_size, writer, epoch, n_iter, val_log)

        torch.cuda.empty_cache()
        if epoch - last_epoch > 100:
            break

        if acc_val_value > current_best_acc:
            current_best_acc = acc_val_value
            torch.save(hmm_model.model.state_dict(), os.path.join(save_dir, 'HMM_model_epoch_{0}.pkl'.format(epoch)))
            torch.save(hmm_model.model.state_dict(), './classify_'+machine_idx+'/HMM_model_' + target_dataset + '_train.pkl')
            val_log['val_log'].info("current best val acc:{0} at epoch {1}".format(current_best_acc, epoch))

    print("Finish!... saved all results")
    writer.close()
    return current_best_acc

if __name__ == '__main__':
    print(str(target_dataset) + 'the best acc is ', main())