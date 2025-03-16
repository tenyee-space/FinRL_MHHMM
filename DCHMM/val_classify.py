from __future__ import print_function
import torch.utils.data
from torchvision.utils import save_image
import numpy as np
import time
import os
from model import Model
from torch import nn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


# args.kl_weight
def kl_loss_function(recon_input, input, mu, logvar, mu_prior, logvar_prior, args):
    BCE = torch.mean((recon_input - input) ** 2)
    KLD = 0.5 * torch.mean(torch.mean(logvar.exp() / logvar_prior.exp() +
                                      (mu - mu_prior).pow(2) / logvar_prior.exp() + logvar_prior - logvar - 1, 1))

    return BCE * args.vae_weight, KLD * args.kl_weight


def val(args, hmm_model: Model, val_dataset, batch_size, writer, epoch, n_iter, val_log):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, drop_last=False, shuffle=False)

    rec_loss_input, kl_loss_input, pre_loss = 0, 0, 0

    hmm_model.eval()
    all_gt_y, all_pre_id = [], []
    with torch.no_grad():
        for i_batch, (sample_batched) in enumerate(val_loader):
            input = torch.tensor(sample_batched[0], device=args.device)
            gt_y = torch.tensor(sample_batched[1], device=args.device)

            # args.h_size
            batch_size = gt_y.shape[0]
            h0 = torch.zeros(batch_size, args.h_dim, device=args.device)
            hmm_model.reset(h0)
            h_t_last = h0
            all_rec_loss, all_kl_loss = torch.zeros(1, device=args.device), torch.zeros(1, device=args.device)
            logit = None
            for index in range(len(input[0])):  # the len of time series
                if index == 0:
                    # input_t = input[:, index]
                    input_t = input
                    input_t_last = torch.zeros_like(input_t)
                    h_t_last = h0
                else:
                    # 修改点 修改格式
                    input_t = input
                    # input_t = input[:, index]
                    input_t_last = input[:, index - 1]
                if input_t.sum() == 0:
                    break
                vae_rec_input_t, mu_h, logvar_h, mu_h_prior, logvar_h_prior \
                    = hmm_model(input_t, test=True)

                loss_re_x_t, loss_kl_x_t = kl_loss_function(vae_rec_input_t, input_t, mu_h, logvar_h, mu_h_prior,
                                                            logvar_h_prior,
                                                            args)
                all_rec_loss += loss_re_x_t
                all_kl_loss += loss_kl_x_t

                if logit is None:
                    logit = hmm_model.classify_one_step()
                else:
                    logit += hmm_model.classify_one_step()

            mean_rec_loss = all_rec_loss / len(input)
            mean_kl_loss = all_kl_loss / len(input)

            rec_loss_input += mean_rec_loss
            kl_loss_input += mean_kl_loss

            writer.add_scalar('val/mean rec loss of image', mean_rec_loss, n_iter)
            writer.add_scalar('val/mean kl loss of image', mean_kl_loss, n_iter)

            writer.add_scalar('val/mean rec loss of image', mean_rec_loss, n_iter)
            writer.add_scalar('val/mean kl loss of image', mean_kl_loss, n_iter)

            func = nn.Softmax(dim=1)  # rescaling
            prob = func(logit)
            prob0 = prob[:, 0].cpu().detach().numpy()
            prob1 = prob[:, 1].cpu().detach().numpy()

            all_gt_y.extend(gt_y.cpu().numpy())  # 按batch维度进行扩充

            all_pre_id.extend(torch.max(logit, 1)[1].data.cpu().numpy())

            cn_loss = nn.CrossEntropyLoss()
            loss_pre = cn_loss(logit, gt_y.long())

            writer.add_scalar('val/classification loss', loss_pre, n_iter)
            pre_loss += loss_pre.item()
            n_iter += 1

    all_gt_y, all_pre_id = \
        np.array(all_gt_y).flatten(), np.array(all_pre_id).flatten()

    auc = accuracy_score(all_gt_y, all_pre_id)

    f1 = f1_score(all_gt_y, all_pre_id, average='micro')

    all_val_acc = (all_pre_id == all_gt_y).sum() / len(all_gt_y)

    writer.add_scalar('val/AUC', auc, epoch)
    writer.add_scalar('val/f1', f1, epoch)
    writer.add_scalar('val/ACC', all_val_acc, epoch)
    val_log['val_log'].info(
        "val acc:{0}, "
        "val auc:{1},val f1:{2}"
        " at epoch {3}"
        "".format(
            all_val_acc,
            auc, f1,
            epoch
        )
    )

    return all_val_acc

def val_with_gt_pre(args, hmm_model: Model, val_dataset, batch_size, writer, epoch, n_iter, val_log):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, drop_last=False, shuffle=False)

    rec_loss_input, kl_loss_input, pre_loss = 0, 0, 0

    hmm_model.eval()
    all_gt_y, all_pre_id = [], []
    with torch.no_grad():
        for i_batch, (sample_batched) in enumerate(val_loader):
            input = torch.tensor(sample_batched[0], device=args.device)
            gt_y = torch.tensor(sample_batched[1], device=args.device)

            # args.h_size
            batch_size = gt_y.shape[0]
            h0 = torch.zeros(batch_size, args.h_dim, device=args.device)
            hmm_model.reset(h0)
            h_t_last = h0
            all_rec_loss, all_kl_loss = torch.zeros(1, device=args.device), torch.zeros(1, device=args.device)
            logit = None
            for index in range(len(input[0])):  # the len of time series
                if index == 0:
                    # input_t = input[:, index]
                    input_t = input
                    input_t_last = torch.zeros_like(input_t)
                    h_t_last = h0
                else:
                    # 修改点 修改格式
                    input_t = input
                    # input_t = input[:, index]
                    input_t_last = input[:, index - 1]
                if input_t.sum() == 0:
                    break
                vae_rec_input_t, mu_h, logvar_h, mu_h_prior, logvar_h_prior \
                    = hmm_model(input_t, test=True)

                loss_re_x_t, loss_kl_x_t = kl_loss_function(vae_rec_input_t, input_t, mu_h, logvar_h, mu_h_prior,
                                                            logvar_h_prior,
                                                            args)
                all_rec_loss += loss_re_x_t
                all_kl_loss += loss_kl_x_t

                if logit is None:
                    logit = hmm_model.classify_one_step()
                else:
                    logit += hmm_model.classify_one_step()

            mean_rec_loss = all_rec_loss / len(input)
            mean_kl_loss = all_kl_loss / len(input)

            rec_loss_input += mean_rec_loss
            kl_loss_input += mean_kl_loss

            writer.add_scalar('val/mean rec loss of image', mean_rec_loss, n_iter)
            writer.add_scalar('val/mean kl loss of image', mean_kl_loss, n_iter)

            writer.add_scalar('val/mean rec loss of image', mean_rec_loss, n_iter)
            writer.add_scalar('val/mean kl loss of image', mean_kl_loss, n_iter)

            func = nn.Softmax(dim=1)  # rescaling
            prob = func(logit)
            prob0 = prob[:, 0].cpu().detach().numpy()
            prob1 = prob[:, 1].cpu().detach().numpy()

            all_gt_y.extend(gt_y.cpu().numpy())  # 按batch维度进行扩充

            all_pre_id.extend(torch.max(logit, 1)[1].data.cpu().numpy())

            cn_loss = nn.CrossEntropyLoss()
            loss_pre = cn_loss(logit, gt_y.long())

            writer.add_scalar('val/classification loss', loss_pre, n_iter)
            pre_loss += loss_pre.item()
            n_iter += 1

    all_gt_y, all_pre_id = \
        np.array(all_gt_y).flatten(), np.array(all_pre_id).flatten()

    auc = accuracy_score(all_gt_y, all_pre_id)

    f1 = f1_score(all_gt_y, all_pre_id, average='micro')

    all_val_acc = (all_pre_id == all_gt_y).sum() / len(all_gt_y)

    writer.add_scalar('val/AUC', auc, epoch)
    writer.add_scalar('val/f1', f1, epoch)
    writer.add_scalar('val/ACC', all_val_acc, epoch)
    val_log['val_log'].info(
        "val acc:{0}, "
        "val auc:{1},val f1:{2}"
        " at epoch {3}"
        "".format(
            all_val_acc,
            auc, f1,
            epoch
        )
    )

    return all_gt_y, all_pre_id