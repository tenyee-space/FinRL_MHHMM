from __future__ import print_function
import torch.utils.data
from torchvision.utils import save_image
import numpy as np
from DCHMM.model import Model
import time
import os

from torch import nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader, Dataset


# args.kl_weight
def kl_loss_function(recon_input, input, mu, logvar, mu_prior, logvar_prior, args):
    BCE = torch.mean((recon_input - input) ** 2)
    KLD = 0.5 * torch.mean(torch.mean(logvar.exp() / logvar_prior.exp() +
                                      (mu - mu_prior).pow(2) / logvar_prior.exp() + logvar_prior - logvar - 1, 1))

    return BCE, KLD


def reset_grad(vae_optimizer):
    """Zeros the gradient buffers."""
    vae_optimizer.zero_grad()


def train(args, hmm_model: Model, train_dataset, batch_size, optimizer_hmm,
          writer, epoch, n_iter, train_log):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, drop_last=True, shuffle=True)

    rec_loss_input, kl_loss_input, pre_loss = 0, 0, 0
    hmm_model.train()
    all_gt_y, all_pre_id = [], []
    for i_batch, (sample_batched) in enumerate(train_loader):  # 第i个batch

        input = torch.tensor(sample_batched[0], device=args.device)
        # print(input)
        gt_y = torch.tensor(sample_batched[1], device=args.device)

        # args.h_size
        batch_size = gt_y.shape[0]
        h0 = torch.zeros(batch_size, args.h_dim, device=args.device)
        hmm_model.reset(h0)
        h_t_last = h0
        all_rec_loss, all_kl_loss = torch.zeros(1, device=args.device), \
                                    torch.zeros(1, device=args.device)
        logit = None
        for index in range(len(input[0])):  # the len of time series
            if index == 0:
                # 修改点 修改格式
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
            # print(f"input_t shape: {input_t.shape}")  # 添加这一行来打印 input_t 的形状
            vae_rec_input_t, mu_h, logvar_h, mu_h_prior, logvar_h_prior \
                = hmm_model(input_t)

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

        writer.add_scalar('train/mean rec loss of image', mean_rec_loss, n_iter)
        writer.add_scalar('train/mean kl loss of image', mean_kl_loss, n_iter)

        all_gt_y.extend(gt_y.cpu().numpy())

        all_pre_id.extend(torch.max(logit, 1)[1].data.cpu().numpy())

        cn_loss = nn.CrossEntropyLoss()
        loss_pre = cn_loss(logit, gt_y.long())

        # update all parameters
        # args.cls_loss_weight
        reset_grad(optimizer_hmm)
        (mean_rec_loss * args.vae_weight + mean_kl_loss * args.kl_weight + loss_pre * args.cls_loss_weight).backward()
        nn.utils.clip_grad_norm(hmm_model.parameters(), max_norm=0.1, norm_type=2)
        optimizer_hmm.step()

        writer.add_scalar('train/classification loss', loss_pre, n_iter)
        pre_loss += loss_pre.item()
        n_iter += 1

    all_gt_y, all_pre_id = \
        np.array(all_gt_y).flatten(), np.array(all_pre_id).flatten()

    auc = accuracy_score(all_gt_y, all_pre_id)

    f1 = f1_score(all_gt_y, all_pre_id, average='micro')

    all_train_acc = (all_pre_id == all_gt_y).sum() / len(all_gt_y)
    writer.add_scalar('train/AUC', auc, epoch)
    writer.add_scalar('train/f1', f1, epoch)
    writer.add_scalar('train/ACC', all_train_acc, epoch)
    print()
    train_log['train_log'].info(
        "train acc:{0}, "
        "kl_loss:{1}, rec_loss:{2},classify loss:{3}"
        " at epoch {4}".format(
            all_train_acc,
            mean_kl_loss.cpu().detach().numpy()[0], mean_rec_loss.cpu().detach().numpy()[0],
            loss_pre,
            epoch
        )
    )

    return n_iter
