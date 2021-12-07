import os
import sys
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
import numpy as np
import random
from datetime import datetime
import logging
import warnings


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def setup_default_logging(args, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    
    output_dir = os.path.join(args.results, args.dataset, f'x{args.n_labeled}_seed{args.seed}', args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('train')
    tmp_timestr = time_str()
    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{tmp_timestr}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)
    return logger, output_dir, tmp_timestr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # return value, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)


class WarmupCosineLrScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))            
            #ratio = 0.5 * (1. + np.cos(np.pi * real_iter / real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


def print_gpu_info():
    print((os.popen("nvidia-smi")).read())
    print(os.environ['CUDA_VISIBLE_DEVICES'])


def process_gpu_args(args):

    if not torch.cuda.is_available():
        return False

    # process gpu_ids
    if isinstance(args.gpu_ids, int):
        args.is_multigpu = False
        args.gpu_list = [args.gpu_ids]
    else:
        if ',' in str(args.gpu_ids):
            args.gpu_ids = args.gpu_ids.strip(",")
        args.gpu_list = [int(x) for x in args.gpu_ids.split(",")]
        if len(args.gpu_list) <= 1:
            args.is_multigpu = False
        else:
            args.is_multigpu = True
    if args.is_multigpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        # RuntimeError: module must have its parameters and buffers on device cuda:2 (device_ids[0]) but found one of them on device: cuda:0
        torch.cuda.set_device('cuda:{}'.format(args.gpu_list[0]))
        print("="*20, args.gpu_ids, args.gpu_list)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
        torch.cuda.set_device('cuda:{}'.format(args.gpu_list[0]))
    args.device = torch.device(f"cuda:{args.gpu_list[0]}")
    # print(args.gpu_list)
    return True
