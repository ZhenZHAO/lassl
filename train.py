from __future__ import print_function
import time
import argparse
import os
import sys

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrn import WideResnet
from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler, set_seed, process_gpu_args, time_str
from torch.utils.tensorboard import SummaryWriter
from config.config import parse_commandline_args, save_config_yaml


# lassl_cifar10.yaml, lassl_svhn.yaml, lassl_cifar100.yaml, lassl_mini.yaml, lassl_cifar10_ablation.yaml
config_file = "./config/lassl_cifar10.yaml" 


def prepare_args(rand_seed):
    description = "Train LaSSL"
    config_path = config_file
    my_args = parse_commandline_args(name=description, filepath=config_path)
    my_args.date_str = time_str()
    if rand_seed is not None:
        my_args.seed = rand_seed
    return my_args


@torch.no_grad()
def _cal_lpa_label(source_feature, source_labels, target_feature, args, num_buffer_low=0):
    eps = torch.finfo(float).eps
    NumS = source_feature.size(0)
    NumT = target_feature.size(0)
    NumST = NumS + NumT

    # hyperparam
    lp_alpha = args.lpa_alpha
    lp_knn_k = int(min(args.lpa_topk, NumT*0.8))
    
    all_feature = torch.cat((source_feature, target_feature), dim=0)  # (Ns + Nt) * d
    target_label_initial = torch.zeros(NumT, args.n_classes).cuda()  # the initial state make no influence
    all_label = torch.cat((source_labels.float(), target_label_initial.float()), dim=0)  # (Ns + Nt) * c

    # knn graph
    weight = torch.exp(torch.mm(all_feature, all_feature.t())/args.temperature)  # N * N
    values, indexes = torch.topk(weight, lp_knn_k, dim=1)
    weight[weight < values[:, -1].view(-1, 1)] = 0
    weight = weight + torch.t(weight)
    weight.diagonal(0).fill_(0)  # zero the diagonal

    # close form solution  # F = (I - \alpha S)^{-1} Y
    D = weight.sum(0)
    D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, NumST)
    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(NumST, 1)
    S = D1 * weight * D2 
#     print(D1.shape, weight.shape, D2.shape, S.shape, S1.shape)
#     PredST = torch.mm(torch.inverse(torch.eye(NumST).cuda() - lp_alpha * S + eps), all_label)
    PredST = torch.mm(torch.inverse(torch.eye(NumST).cuda() - lp_alpha * S + 1e-7), all_label) 

    res = PredST[NumS+num_buffer_low:, :]

    return res / res.sum(dim=1).view(-1, 1)


@torch.no_grad()
def get_pseudo_label_propagation_wt_buffer(cur_features, cur_labels,
    buffer_features, buffer_labels, target_feature, args, it):
    cur_labels = cur_labels.float()
    if it < 1 or args.bootstrap_nums == 0:
        source_feature = cur_features
        source_labels = cur_labels
        num_buffer_low = 0
        return _cal_lpa_label(source_feature, source_labels, target_feature, args, num_buffer_low=num_buffer_low)

    if not args.use_buffer_bootstrap or args.bootstrap_nums == 1:
        v_max, i_max = torch.max(buffer_labels, dim=-1)
        mask = i_max.ge(args.bootstrap_thred)
        buffer_features_high = buffer_features[mask]
        buffer_labels_high = buffer_labels[mask]
        buffer_features_low = buffer_features[~mask]
        
        source_feature = torch.cat((cur_features, buffer_features_high), dim=0)
        source_labels = torch.cat((cur_labels, buffer_labels_high), dim=0)

        target_feature = torch.cat((buffer_features_low, target_feature), dim=0)
        num_buffer_low = buffer_features_low.shape[0]

        return _cal_lpa_label(source_feature, source_labels, target_feature, args, num_buffer_low=num_buffer_low)
    
    res = []
    for _ in range(args.bootstrap_nums):
        shuffle_indexes = np.random.permutation(len(buffer_features))
        ratio_index = int(len(buffer_features) * args.bootstrap_ratio)
        select_buffer_features = buffer_features[shuffle_indexes[:ratio_index]]
        select_buffer_labels = buffer_labels[shuffle_indexes[:ratio_index]]
        
        v_max, i_max = torch.max(select_buffer_labels, dim=-1)
        mask = i_max.ge(args.bootstrap_thred)

        buffer_features_high = select_buffer_features[mask]
        buffer_labels_high = select_buffer_labels[mask]
        buffer_features_low = select_buffer_features[~mask]

        source_feature = torch.cat((cur_features, buffer_features_high), dim=0)
        source_labels = torch.cat((cur_labels, buffer_labels_high), dim=0)
        target_feature_new = torch.cat((buffer_features_low, target_feature), dim=0)
        num_buffer_low = buffer_features_low.shape[0]

        tmp_res = _cal_lpa_label(source_feature, source_labels, target_feature_new, args, num_buffer_low=num_buffer_low)
        res.append(tmp_res)
    
    tensor_res = torch.stack(res, dim=0)    
    return tensor_res.mean(dim=0)


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)    


@torch.no_grad()
def get_quantity_quality_of_pseudo(gt_labels, pseudo_labels, confidence_thres):
    y_u_max_LPA, y_u_LPA = torch.max(pseudo_labels, dim=-1)
    mask_LPA = y_u_max_LPA.ge(confidence_thres).float()
    num_correct_pseudo_labels_LPA = (y_u_LPA == gt_labels).float() * mask_LPA

    num_high_pseudo_ratio = mask_LPA.mean().item()
    num_correct = num_correct_pseudo_labels_LPA.sum().item()
    num_pseudo_total = mask_LPA.sum().item()

    return num_high_pseudo_ratio, num_pseudo_total, num_correct


@torch.no_grad()
def get_sharpen_label(pseudo, var_temp):  # larger var_temp --> smooth/flat, small var_temp --> sharp
    pt = pseudo**(1/var_temp)
    return pt / pt.sum(dim=1, keepdim=True)


def exp_rampdown(num_epochs, rampdown_length=None, delta=0.5):
    if rampdown_length is None:
        rampdown_length = int(num_epochs * 0.9)
    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = delta * (epoch - (num_epochs - rampdown_length))
            return float(np.exp(-(ep * ep) / rampdown_length))
        else:
            return 1.0
    return warpper


def train(epoch, model, ema_model, criteria_x, optim, lr_schdlr, dltrain_x, dltrain_u, 
            args, n_iters,logger, dist_class):
    # buffer for caching the last iteration
    buffer_features = torch.zeros(int(args.batchsize * (args.mu + 1)), args.low_dim).cuda() # non_blocking=True
    buffer_labels = torch.zeros(int(args.batchsize * (args.mu + 1)), args.n_classes).cuda() # non_blocking=True

    # weight of loss_c
    weight_loss_c = exp_rampdown(args.n_epoches, int(args.n_epoches - args.rampdown_fix_len), args.rampdown_delta)(epoch)

    # output new distribution used for MDA
    cur_pred_dist = None

    # losses
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    loss_c_meter = AverageMeter()
    # debugging: quality and quantity of pseudo-label
    n_correct_u_lbs_meter = AverageMeter()
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    
    # dist
    if dist_class is not None:
        dist_gt = dist_class.cuda()
    
    # train
    model.train()
    epoch_start = time.time()
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters):
        (ims_u_weak, ims_u_strong), lbs_u_real = next(dl_u)
        (ims_x_weak, ims_x_strong), lbs_x = next(dl_x)
        lbs_x = lbs_x.cuda() # cuda(non_blocking=True)
        lbs_u_real = lbs_u_real.cuda() #cuda(non_blocking=True)
        #------------ 1) inference phase
        bt = ims_x_weak.size(0)
        btu = ims_u_weak.size(0)
        imgs = torch.cat([ims_x_weak, ims_x_strong, ims_u_weak, ims_u_strong], dim=0).cuda() # cuda(non_blocking=True)
        logits, features = model(imgs)

        logits_x_w, logits_x_s = torch.split(logits[:(bt+bt)], bt)
        logits_u_w, logits_u_s = torch.split(logits[(bt+bt):], btu)
        feats_x_w, feats_x_s = torch.split(features[:(bt+bt)], bt)
        feats_u_w, feats_u_s = torch.split(features[(bt+bt):], btu)
        # no backpropagation ---> detach
        logits_u_w = logits_u_w.detach() 
        feats_x_w = feats_x_w.detach()
        feats_u_w = feats_u_w.detach()
        # revise pseudo-laebls
        with torch.no_grad():
            # network labels
            probs = torch.softmax(logits_u_w, dim=1)
            # MDA
            if args.MDA:
                tmp_dist = probs.mean(0)
                # update pred_dist
                if cur_pred_dist is None:
                    if dist_class is not None:
                        cur_pred_dist = dist_gt.clone()
                    else:
                        cur_pred_dist = torch.ones_like(tmp_dist) / args.n_classes
                cur_pred_dist = cur_pred_dist * args.mda_hist_mom + tmp_dist * (1 - args.mda_hist_mom)
                
                if dist_class is not None:
                    prob = probs * dist_gt / cur_pred_dist
                else:
                    probs = probs / cur_pred_dist
                probs = probs / probs.sum(dim=1, keepdim=True)
            
            # LPA
            if args.blpa_join_early and args.blpa_join_later:
                if weight_loss_c > args.rampdown_lpa_thr:
                    BLPA_join = True
                else:
                    BLPA_join = False
            elif args.blpa_join_early:
                if weight_loss_c > 0.99:
                    BLPA_join = True
                else:
                    BLPA_join = False
            elif args.blpa_join_later:
                if weight_loss_c > args.rampdown_lpa_thr and weight_loss_c < 0.99:
                    BLPA_join = True
                else:
                    BLPA_join = False
            else:
                BLPA_join = False
            
            if epoch < 1:
                tmp_label_alpha = 0
            else:
                if args.lpa_ramp_down:
                    tmp_label_alpha = args.embedding_pseudo_ratio * weight_loss_c
                else:
                    tmp_label_alpha = args.embedding_pseudo_ratio
            if args.BLPA and BLPA_join:
                pseudo_label_propagation = get_pseudo_label_propagation_wt_buffer(
                    feats_x_w, torch.nn.functional.one_hot(lbs_x.detach(), num_classes=args.n_classes), 
                    buffer_features, buffer_labels, feats_u_w, args, it)
                pseudo_label = (1.0 - tmp_label_alpha) * probs + tmp_label_alpha * pseudo_label_propagation
                
                # update buffer of latest batch
                buffer_features = torch.cat((feats_x_w, feats_u_w), dim=0)
                # buffer_labels = torch.cat((torch.nn.functional.one_hot(lbs_x.detach(), num_classes=args.n_classes).float(), pseudo_label.float()), dim=0)
                buffer_labels = torch.cat((torch.nn.functional.one_hot(lbs_x.detach(), num_classes=args.n_classes).float(), probs.float()), dim=0)
            else:
                pseudo_label = probs

            # mask of high-confidence
            scores, lbs_u_guess = torch.max(pseudo_label, dim=1)
            mask_bool = scores.ge(args.threshold)
            mask = mask_bool.float()
            # final pseudo label
            pseudo_label = pseudo_label.detach()

        #------------ 2) training phase
        # a) label loss
        loss_x = criteria_x(logits_x_w, lbs_x) 
        # b) unsupervised loss
        if any(mask_bool):
            # unlabel
            loss_u = - torch.sum((F.log_softmax(logits_u_s,dim=1) * pseudo_label), dim=1) * mask
            loss_u = loss_u.mean()

            # contrastive
            if weight_loss_c < args.rampdown_cacl_thr:
                args.CACL = False
            if args.CACL:
                cont_features = torch.cat([feats_x_s, feats_u_s])
                cont_prob = torch.cat([(torch.nn.functional.one_hot(lbs_x.detach(), num_classes=args.n_classes)).float(), pseudo_label])

                sim = torch.exp(torch.mm(cont_features, cont_features.t())/args.temperature) 
                sim_probs = sim / sim.sum(1, keepdim=True)

                # instance relation graph
                Q = torch.mm(cont_prob, cont_prob.t()) 
                Q.fill_diagonal_(1)    
                pos_mask = (Q >= args.contrast_th).float()
                Q = Q * pos_mask
                Q = Q / Q.sum(1, keepdim=True)
                loss_c = - (torch.log(sim_probs + 1e-6) * Q).sum(1)
                loss_c = loss_c.mean()
            else:
                loss_c = torch.tensor(0.0).cuda()
        else:
            loss_u = torch.tensor(0.0).cuda()
            loss_c = torch.tensor(0.0).cuda()

        # total loss
        loss = loss_x + args.lambda_semi * loss_u + args.lambda_cont * weight_loss_c * loss_c
        #------------ 3) Updating
        # updates weights
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()
        if args.use_ema_model:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)        

        # tracking quality and quantity
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        loss_c_meter.update(loss_c.item())
        mask_meter.update(mask.mean().item())
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())
        
        # printings
        if (it + 1) % args.print_freq == 0:
            t = time.time() - epoch_start
            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)
            tmp_accuracy = n_correct_u_lbs_meter.avg / n_strong_aug_meter.avg if n_strong_aug_meter.avg > 0 else 0
            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_x: {:.3f}. loss_u: {:.3f}. loss_c:{:.3f}. "
                        "n_correct_u: {:.2f}/{:.2f}. Mask:{:.3f}. C-Ratio:{:.3f}. LR: {:.3f}. Time: {:.2f} Weight:{:.3f}".format(
                args.dataset, args.n_labeled, args.seed, args.model_name, epoch, it + 1, loss_x_meter.avg, loss_u_meter.avg, loss_c_meter.avg,
                n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, tmp_accuracy, lr_log, t, weight_loss_c))

            epoch_start = time.time()
    avg_accuracy = n_correct_u_lbs_meter.avg / n_strong_aug_meter.avg if n_strong_aug_meter.avg > 0 else 0
    return loss_x_meter.avg, loss_u_meter.avg, loss_c_meter.avg, mask_meter.avg, n_correct_u_lbs_meter.avg, avg_accuracy


def test(model, ema_model, dataloader):
    model.eval()
    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()
    if ema_model is not None:
        ema_model.eval()
    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda() # cuda(non_blocking=True)
            lbs = lbs.cuda() # cuda(non_blocking=True)

            logits, _ = model(ims)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            
            if ema_model is not None:
                logits, _ = ema_model(ims)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))                
                ema_top1_meter.update(top1.item())

    return top1_meter.avg, ema_top1_meter.avg


def set_model(args):
    model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        msg = model.load_state_dict(checkpoint, strict=False)
        assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
        print('loaded from checkpoint: %s'%args.checkpoint)    
    model.train()
    model.cuda()
    if args.is_multigpu:
        model = nn.DataParallel(model, device_ids=args.gpu_list, output_device=args.gpu_list[0])

    if args.use_ema_model:
        ema_model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=True)
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False
        ema_model.cuda()  
        ema_model.eval()
        if args.is_multigpu:
            ema_model = nn.DataParallel(ema_model, device_ids=args.gpu_list, output_device=args.gpu_list[0])
    else:
        ema_model = None

    criteria_x = nn.CrossEntropyLoss().cuda()
    return model,  criteria_x, ema_model


def main(my_args):
    args = my_args
    if args.seed > 0:
        set_seed(args.seed)
    if not process_gpu_args(args):  # GPU
        print("Please using GPUs for training")
        return
    if args.is_multigpu:
        torch.cuda.device_count()
        gpu_num = len(args.gpu_list)
        # using dataparallel which will directly divided the batch size
        # args.batchsize *= gpu_num
    if not args.CACL:
        args.BLPA = False

    ###########################
    # 1. output settings
    ###########################
    logger, output_dir, curr_timestr = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))
    csv_path = os.path.join(output_dir, "{}_stat.csv".format(curr_timestr))
    tb_logger = SummaryWriter(output_dir)

    ###########################
    # 2. prepare data
    ###########################
    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  # 1024
    n_iters_all = n_iters_per_epoch * args.n_epoches

    logger.info("======================== Running training ========================")
    logger.info(f"  Train >> Dataset:{args.dataset} labeledNum:{args.n_labeled} Epoches:{args.n_epoches} Seed:{args.seed}")
    if args.dataset.startswith("CIFAR"):
        from datasets.cifar import get_train_loader, get_val_loader
        dltrain_x, dltrain_u = get_train_loader(
            args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, num_workers=args.workers,
            root=args.root, method=args.label_aug, unmethod=args.unlabel_aug)
        dlval = get_val_loader(dataset=args.dataset, batch_size=args.batchsize, num_workers=args.workers, root=args.root)
        class_dist = None
    elif args.dataset.startswith("SVHN"):
        from datasets.svhn import get_train_loader, get_val_loader
        dltrain_x, dltrain_u, class_dist = get_train_loader(
            args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, num_workers=args.workers,
            root=args.root, method=args.label_aug, unmethod=args.unlabel_aug)
        dlval = get_val_loader(dataset=args.dataset, batch_size=args.batchsize, num_workers=args.workers, root=args.root)

    ###########################
    # 3. prepare model
    ###########################
    model, criteria_x, ema_model = set_model(args)
    logger.info("  Model >> Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    
    ##############################
    # 4. optimizer & scheduler
    ##############################
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)  
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    ######################################
    # 7. training loop
    ######################################
    train_args = dict(
        model=model, ema_model=ema_model, criteria_x=criteria_x,
        optim=optim, lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x, dltrain_u=dltrain_u,
        args=args, n_iters=n_iters_per_epoch,
        logger=logger, dist_class=class_dist,
    )
    best_acc, best_acc_ema = -1, -1
    best_epoch, best_epoch_ema = 0,0
    logger.info("  Ablation >> CACL:{}. BLPA:{}. MDA:{}.".format(args.CACL, args.BLPA, args.MDA))
    logger.info("  Temp-debug >> CACL-len:{}. ramp-down:{}/{}. LPA_label_ratio:{}-rampdown-{}".format(args.rampdown_fix_len, 
        args.rampdown_delta, args.rampdown_lpa_thr, args.embedding_pseudo_ratio, args.lpa_ramp_down))
    logger.info(f"  Weight-Debug >> Join-early:{args.blpa_join_early}. Join-late:{args.blpa_join_later}. label-aug:{args.label_aug}.")
    logger.info('-------------------------- start training --------------------------')
    for epoch in range(args.n_epoches):
        # training
        loss_x, loss_u, loss_c, mask_mean, acc_num, pseudo_label_acc = train(epoch, **train_args)
        # testing
        top1, ema_top1 = test(model, ema_model, dlval)
        
        # best acc
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        if best_acc_ema < ema_top1:
            best_acc_ema = ema_top1
            best_epoch_ema = epoch

        # save tensorboard
        tb_logger.add_scalar('labeled_loss', loss_x, epoch)
        tb_logger.add_scalar('semi_loss', loss_u, epoch)
        tb_logger.add_scalar('cont_loss', loss_c, epoch)
        tb_logger.add_scalar('acc_num', acc_num, epoch)
        tb_logger.add_scalar('quantity', mask_mean, epoch)
        tb_logger.add_scalar('quality', pseudo_label_acc, epoch)
        tb_logger.add_scalar('test_acc', top1, epoch)
        tb_logger.add_scalar('test_ema_acc', ema_top1, epoch)

        # save statistics
        tmp_results = {'labeled_loss': loss_x,
                       'semi_loss': loss_u,
                       'cont_loss': loss_c,
                       "acc_num": acc_num,
                       'quantity': mask_mean,
                       'quality': pseudo_label_acc,
                       'test_acc': top1, 
                       "test_acc_ema": ema_top1}
        data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch+1))
        if epoch > 0:
            data_frame.to_csv(csv_path, mode='a', header=None, index_label='epoch')
        else:
            data_frame.to_csv(csv_path, index_label='epoch')
        
        # print:
        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f}/{}. best_acc_ema: {:.4f}/{}.".
                    format(epoch, top1, ema_top1, best_acc, best_epoch, best_acc_ema, best_epoch_ema))
    logger.info(f"======== best_acc:{best_acc}/{best_acc_ema}")
    return max(best_acc, best_acc_ema)


if __name__ == '__main__':
    flag_run_once = True
    if flag_run_once:
        run_args = prepare_args(None)
        best_score = main(run_args)
        out_str = f"{run_args.seed}: {best_score}: {run_args.date_str}"
    else:
        random_seeds = [1, 2, 3, 4, 5]
        best_scores = []
        time_strs = []
        for each_seed in random_seeds:
            run_args = prepare_args(each_seed)
            time_strs.append(run_args.date_str)
            best_scores.append(main(run_args))
        out_str = "\n".join([f"{x}: {z} : {y}" for x, y, z in zip(random_seeds, time_strs, best_scores)])
    
    print("==="*24, f"\n{out_str}")
