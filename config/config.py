import argparse
import os
import yaml


def save_config_yaml(var_dict, file_path):
    if "device" in var_dict.keys():
        # del var_dict["device"]
        var_dict.pop('device')
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(var_dict, f, default_flow_style=False, encoding='utf-8', allow_unicode=True)


def yaml_config_hook(config_file):
    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def create_parser_from_file(name, filepath="./lassl.yaml"):
    parser = argparse.ArgumentParser(description=name)
    config = yaml_config_hook(filepath)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser


def create_parser(name):
    parser = argparse.ArgumentParser(description=name) 
    parser.add_argument('--gpu-ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--workers', default=2, type=int, help='multiprocessing')
    
    # directory
    parser.add_argument('--root', default='../data/cifar10/', type=str, help='dataset directory')
    parser.add_argument('--results', default='./results', type=str, help='results directory')
    parser.add_argument('--model-name', default='LaSSL', type=str, help='Name of SSL method')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    
    # training options
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument('--print-freq', type=int, default=128, help='the freq of printing info')

    # optim and scheduler    
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')

    # model
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--use-ema-model', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)

    # dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset name')
    parser.add_argument('--n-classes', type=int, default=10,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training. balanced by default')
    
    # ----------------------- Pseudo label
    parser.add_argument('--lambda-semi', type=float, default=1., help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', type=float, default=0.95, help='pseudo label threshold')   

    # ----------------------- CACL
    parser.add_argument('--low-dim', type=int, default=64)
    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--contrast-th', default=0.8, type=float, help='instance relationship threshold')       

    # ----------------------- BLPA
    parser.add_argument('--blpa_join_early', default=False, help='to use BLPA at the begining or later')
    parser.add_argument('--lpa_alpha', type=float, default=0.8, help='alpha for lpa closed solution')
    parser.add_argument('--lpa_topk', type=int, default=15, help='K nearest neighbors')

    parser.add_argument('--use_buffer_bootstrap', default=True, help='whether to use buffer-aided')
    parser.add_argument('--bootstrap_thred', type=float, default=0.95, help='high-confidence threshold for blpa')
    parser.add_argument('--bootstrap_nums', type=int, default=3, help='sampling times') 
    parser.add_argument('--bootstrap_ratio', type=float, default=0.8, help='sampling ratio')
    
    parser.add_argument('--embedding_pseudo_ratio', type=float, default=0.2, help='ratio for blpa pseudo-label')
    parser.add_argument('--lpa_ramp_down', default=False, help='decay the ratio of blpa pseudo-label')

    # -----------------------  Weights, MDA
    parser.add_argument('--lambda_cont', type=float, default=1.0, help='coefficient of contrastive loss')
    parser.add_argument('--rampdown_fix_len', type=int, default=30, help='initial length without loss weight decay') 
    parser.add_argument('--rampdown_delta', type=float, default=1.0, help='xxx') #  the larger, fast converge to 0.
    parser.add_argument('--rampdown_lpa_thr', type=float, default=0.3, help='stop blpa when weight is small ')
    parser.add_argument('--rampdown_cacl_thr', type=float, default=0.1, help='stop calcl when weight is small ')
    parser.add_argument('--mda_hist_mom', type=float, default=0.99, help='momentum paramer for MDA')

    # ----------------------- Augmentations
    parser.add_argument('--label_aug', default='self_weak', type=str, help='augmentations for labeled data')
    parser.add_argument('--unlabel_aug', default='semi', type=str, help='augmentations for unlabeled data')

    # ----------------------- Ablation
    parser.add_argument('--CACL', default=True, help='use CACL')
    parser.add_argument('--MDA', default=False, help='use MDA')
    parser.add_argument('--BLPA', default=True, help='use BLPA')

    return parser


def parse_commandline_args(name="LaSSL", filepath="./lassl.yaml"):
    '''
        args = parser.parse_args('')  # running in ipynb
        args = parser.parse_args()  # running in command line
    '''
    if filepath is not None and os.path.exists(filepath) and not os.path.isdir(filepath):
        args = create_parser_from_file(name, filepath).parse_args()
    else:
        args = create_parser(name).parse_args()
    return args


if __name__ == '__main__':
    test_parse_file = True
    if test_parse_file:
        my_args = parse_commandline_args()
        print("=========parse from file=======")
    else:
        my_args = parse_commandline_args(filepath=None)
        print("=========argparse CLI=======")
    
    # show info
    for key in my_args.__dict__:
        var = my_args.__dict__[key]
        v_type = type(var)
        print(f"{key}:{var}:{v_type}")
