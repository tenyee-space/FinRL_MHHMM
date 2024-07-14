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

warnings.filterwarnings('ignore')

dataset_list = {
    'AWR': {'name': 'ArticularyWordRecognition', 'train_size': '275', 'test_size': '300', 'dims': '9', 'length': '144',
            'classes': '25', 'batch_size': '24'},
    'AF': {'name': 'AtrialFibrillation', 'train_size': '15', 'test_size': '15', 'dims': '2', 'length': '640',
           'classes': '3', 'batch_size': '1'},
    'BM': {'name': 'BasicMotions', 'train_size': '40', 'test_size': '40', 'dims': '6', 'length': '100', 'classes': '4',
           'batch_size': '4'},
    'CT': {'name': 'CharacterTrajectories', 'train_size': '1422', 'test_size': '1436', 'dims': '3', 'length': '182',
           'classes': '20', 'batch_size': '64'},
    'CR': {'name': 'Cricket', 'train_size': '108', 'test_size': '72', 'dims': '6', 'length': '1197', 'classes': '12',
           'batch_size': '8'},
    'DDG': {'name': 'DuckDuckGeese', 'train_size': '50', 'test_size': '50', 'dims': '270', 'length': '270',
            'classes': '5', 'batch_size': '5'},
    'EW': {'name': 'EigenWorms', 'train_size': '128', 'test_size': '131', 'dims': '6', 'length': '17984',
           'classes': '5', 'batch_size': '16'},
    'EP': {'name': 'Epilepsy', 'train_size': '137', 'test_size': '138', 'dims': '3', 'length': '206', 'classes': '4',
           'batch_size': '8'},
    'ER': {'name': 'EthanolConcentration', 'train_size': '261', 'test_size': '263', 'dims': '3', 'length': '1751',
           'classes': '4', 'batch_size': '32'},
    'EC': {'name': 'ERing', 'train_size': '30', 'test_size': '270', 'dims': '4', 'length': '65', 'classes': '6',
           'batch_size': '3'},
    'FD': {'name': 'FaceDetection', 'train_size': '5890', 'test_size': '3524', 'dims': '144', 'length': '62',
           'classes': '2', 'batch_size': '256'},
    'FM': {'name': 'FingerMovements', 'train_size': '316', 'test_size': '100', 'dims': '28', 'length': '50',
           'classes': '2', 'batch_size': '32'},
    'HMD': {'name': 'HandMovementDirection', 'train_size': '160', 'test_size': '74', 'dims': '10', 'length': '400',
            'classes': '4', 'batch_size': '16'},
    'HW': {'name': 'Handwriting', 'train_size': '150', 'test_size': '850', 'dims': '3', 'length': '152',
           'classes': '26', 'batch_size': '16'},
    'HB': {'name': 'Heartbeat', 'train_size': '204', 'test_size': '205', 'dims': '61', 'length': '405', 'classes': '2',
           'batch_size': '24'},
    'IW': {'name': 'InsectWingbeat', 'train_size': '25000', 'test_size': '25000', 'dims': '200', 'length': '30',
           'classes': '10', 'batch_size': '2048'},
    'JV': {'name': 'JapaneseVowels', 'train_size': '270', 'test_size': '370', 'dims': '12', 'length': '29',
           'classes': '9', 'batch_size': '24'},
    'LIB': {'name': 'Libras', 'train_size': '180', 'test_size': '180', 'dims': '2', 'length': '45', 'classes': '15',
            'batch_size': '16'},
    'LSST': {'name': 'LSST', 'train_size': '2459', 'test_size': '2466', 'dims': '6', 'length': '36', 'classes': '14',
             'batch_size': '256'},
    'MI': {'name': 'MotorImagery', 'train_size': '278', 'test_size': '100', 'dims': '64', 'length': '3000',
           'classes': '2', 'batch_size': '24'},
    'NATO': {'name': 'NATOPS', 'train_size': '180', 'test_size': '180', 'dims': '24', 'length': '51', 'classes': '6',
             'batch_size': '16'},
    'PD': {'name': 'PenDigits', 'train_size': '7494', 'test_size': '3498', 'dims': '2', 'length': '8', 'classes': '10',
           'batch_size': '768'},
    'PEMS': {'name': 'PEMS-SF', 'train_size': '267', 'test_size': '173', 'dims': '963', 'length': '144', 'classes': '7',
             'batch_size': '24'},
    'PS': {'name': 'PhonemeSpectra', 'train_size': '3315', 'test_size': '3353', 'dims': '11', 'length': '217', 'classes': '39',
           'batch_size': '320'},
    'RS': {'name': 'RacketSports', 'train_size': '151', 'test_size': '152', 'dims': '6', 'length': '30', 'classes': '4',
           'batch_size': '16'},
    'SRS1': {'name': 'SelfRegulationSCP1', 'train_size': '268', 'test_size': '293', 'dims': '6', 'length': '896',
             'classes': '2', 'batch_size': '24'},
    'SRS2': {'name': 'SelfRegulationSCP2', 'train_size': '200', 'test_size': '180', 'dims': '7', 'length': '1152',
             'classes': '2', 'batch_size': '24'},
    'SAD': {'name': 'SpokenArabicDigits', 'train_size': '6599', 'test_size': '2199', 'dims': '13', 'length': '93',
            'classes': '10', 'batch_size': '768'},
    'SWJ': {'name': 'StandWalkJump', 'train_size': '12', 'test_size': '15', 'dims': '4', 'length': '2500',
            'classes': '3', 'batch_size': '1'},
    'UWGL': {'name': 'UWaveGestureLibrary', 'train_size': '120', 'test_size': '320', 'dims': '3', 'length': '315',
             'classes': '8', 'batch_size': '12'},
     'SPI': {'name': 'SharePriceIncrease', 'train_size': '965', 'test_size': '965', 'dims': '1', 'length': '60',
              'classes': '2', 'batch_size': '1'}  }


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

parser1 = argparse.ArgumentParser(description='pre-input')
parser1.add_argument('--target-dataset', type=str, default='SPI', help='None')
parser1.add_argument('--machine-idx', type=str, default='v1', help='None')
parser1.add_argument('--v-dims', type=int, default=128, help='None')
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
parser.add_argument('--k', type=int, default=5, help='None')
parser.add_argument('--m', type=int, default=3, help='None')
parser.add_argument('--class_num', type=int, default=dataset_list[target_dataset]['classes'], help='None')
parser.add_argument('--input_fc_dim', type=int, default=v_dims, help='input_fc_dim')
parser.add_argument('--h-dim', type=int, default=v_dims, metavar='h', help='latent size of h in vae (default: 128)')

parser.add_argument('--device', type=str, default='cuda:0', help='None')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.0001)')
parser.add_argument('--epoch', type=int, default=400, help='training epoch')
parser.add_argument('--batch-size', type=int, default=dataset_list[target_dataset]['batch_size'], help='batch size')
parser.add_argument('--init', default='xavier', help='default, this can not be optional')
parser.add_argument('--optimizer', default='Adam', help='optimizer (Adam or RMSprop, SGD, Adagrad, Momentum, Adadelta)')
parser.add_argument('--kl-weight', type=float, default=10, metavar='LR', help='KL loss weight (default: 0.1)')
parser.add_argument('--vae-weight', type=float, default=1, metavar='LR', help='VAE loss weight (default: 0.1)')
parser.add_argument('--cls-loss-weight', type=float, default=10, metavar='LR',
                    help='classification loss weight (default: 0.1)')
parser.add_argument('--cls-fc-dim', type=int, default=v_dims, help='cls FC dim')
parser.add_argument('--load-pre-train-weight', type=bool, default=True,
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
    hmm_model = Model(args)
    hmm_model.init_parameters()
    print('HMM Model', hmm_model)
    if args.load_pre_train_weight:
        print('load pre-train weight')
        print(f'the pre-train weight\'s path is {args.pre_train_weight}')

        weight = hmm_model.state_dict()
        try:
            per_train_weight = torch.load(args.pre_train_weight)
            hmm_model.load_state_dict(load_pretrained(per_train_weight, weight))
            del per_train_weight
        except:
            print('without pre-train model, load nothing')

    lr = args.lr

    optimizer_hmm = optim.Adam(hmm_model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
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

    train_set_path = args.dataset_dir + '/' + args.datasetname + '/' + args.datasetname + '_TRAIN.csv'
    train_label_path = args.dataset_dir + '/' + args.datasetname + '/train_label.csv'
    val_set_path = args.dataset_dir + '/' + args.datasetname + '/' + args.datasetname + '_TEST.csv'
    val_label_path = args.dataset_dir + '/' + args.datasetname + '/test_label.csv'

    train_dataset = load_dataset(args, train_set_path, train_label_path, train=True)
    val_dataset = load_dataset(args, val_set_path, val_label_path, train=False)

    mi1, ma1 = train_dataset.get_min_max()
    mi2, ma2 = val_dataset.get_min_max()
    mi = min(mi1, mi2)
    ma = max(ma1, ma2)
    print(mi, ma)
    del mi1, mi2, mi, ma1, ma2, ma

    train_dataset.normal_data()
    val_dataset.normal_data()

    train_dataset.print_features()
    val_dataset.print_features()

    print('train num', train_dataset.__len__(), 'val num',
          val_dataset.__len__())

    n_iter = 0
    current_best_acc = 0
    last_epoch = 0

    for epoch in range(train_epoch):
        print('training in epoch {0}'.format(epoch))
        n_iter = train(args, hmm_model, train_dataset, batch_size, optimizer_hmm, writer, epoch, n_iter, train_log)

        scheduler_hmm.step()

        acc_val_value = val(args, hmm_model, val_dataset, batch_size, writer, epoch, n_iter, val_log)

        torch.cuda.empty_cache()
        if epoch - last_epoch > 100:
            break
        
        if acc_val_value > current_best_acc:
            current_best_acc = acc_val_value
            torch.save(hmm_model.state_dict(), os.path.join(save_dir, 'HMM_model_epoch_{0}.pkl'.format(epoch)))
            torch.save(hmm_model.state_dict(), './classify_'+machine_idx+'/HMM_model_' + target_dataset + '_train.pkl')
            val_log['val_log'].info("current best val acc:{0} at epoch {1}".format(current_best_acc, epoch))

    print("Finish!... saved all results")
    writer.close()
    return current_best_acc


if __name__ == '__main__':
    print(str(target_dataset) + 'the best acc is ', main())
