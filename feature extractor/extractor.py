import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import model, utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
import UCF_Crime_DVS
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from spikingjelly.datasets import pad_sequence_collate
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import autoaugment
import h5py
import torch
import cv2
import scipy.ndimage
# from metavision_ml.data.sequential_dataset import SequentialDataset
# from metavision_ml.data.sequential_dataset import SequentialDataLoader
from torch.utils.data import Dataset, DataLoader
_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
writer = SummaryWriter("./")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='SEWResNet', help='model')
    parser.add_argument('--dataset', default='UCF-Crime_DVS', help='dataset')
    parser.add_argument('--num-classes', type=int, default=14, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='/media/qyb/My Passport/UCFC-DVS_raw', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='/home/qyb/qyb/SnnAd/checkpoint/checkpoint-284.pth.tar', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')

    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--dim', type=int, default=None, metavar='N',
                        help='embedding dimsension of feature')
    parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                        help='attention head number')
    parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                        help='Image patch size')
    parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                        help='expand ration of embedding dimension in MLP block')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=192, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--nproc_per_node', default=0, type=int,
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    args = parser.parse_args()
    return args


def extract_features(model, data_loader, output_dir,device, print_freq=100, header='Feature Extraction:'):
    model.eval()

    features = []  
    metric_logger = utils.MetricLogger(delimiter="  ")
    feature_count = 0
    with torch.no_grad():
        feature_count = 0
        target_counts = {target: 0 for target in range(14)}  
        for image, target, file_paths,*rest in metric_logger.log_every(data_loader, print_freq, header):
            # image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # image = image.float()
            N, T, C, H, W = image.shape

            # 如果T大于1000,则将其分成多个小块
            if T > 100:
                split_size = 100
                num_splits = T // split_size
                if T % split_size != 0:
                    num_splits += 1

                all_features = []
                for i in range(num_splits):
                    start = i * split_size
                    end = min((i + 1) * split_size, T)
                    split_image = image[:, start:end]
                    split_image = split_image.to(device,non_blocking=True).float()
                    split_feature = model(split_image)
                    functional.reset_net(model)
                    all_features.append(split_feature.cpu())

                feature = torch.cat(all_features, dim=0)
            else:
                image = image.to(device, non_blocking=True).float()
                feature = model(image)
                functional.reset_net(model)
                feature = feature.cpu()

            for i in range(N):
                file_path = rest[i]  
                feature_path = os.path.join(output_dir,
                                            f"{os.path.splitext(os.path.basename(file_path[0]))[0]}_feature.npy")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                np.save(feature_path, feature[i].numpy())




    return features


def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    # origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')
    # dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)
    dataset_train = UCF_Crime_DVS.UCF_Crime_DVS(root=dataset_dir, train=True, data_type='frame', frames_number=T,
                                                 split_by='number')
    # dataset_train = UCF_Crime_DVS.UCF_Crime_DVS(root=dataset_dir, train=True, data_type='frame',duration=533328)
    dataset_test = UCF_Crime_DVS.UCF_Crime_DVS(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                split_by='number')
    # dataset_test = UCF_Crime_DVS.UCF_Crime_DVS(root=dataset_dir, train=False, data_type='frame', duration=533328)
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler



def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    data_path = args.data_path

    # data_loader_test = load_data(data_path, args.T,args.batch_size,args.workers)
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)
    # dataset_test, test_sampler = load_data(data_path, args.distributed, args.T)
    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
        collate_fn=pad_sequence_collate,
        # sampler=train_sampler,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        # sampler=test_sampler,
        drop_last=False,
        collate_fn=pad_sequence_collate,
        pin_memory=True)

    model = create_model(
        'Spikingformer',
         pretrained=False,
         drop_rate=0.,
         drop_path_rate=0.1,
         drop_block_rate=None,
    )

    # model = create_model(
    #     'Spikingformer',
    #     pretrained=False,
    #     drop_rate=0.,
    #     drop_path_rate=0.1,
    #     drop_block_rate=None,
    #     img_size_h=224, img_size_w=224,
    #     patch_size=args.patch_size,  num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
    #
    #     # depths=args.depths, sr_ratios=1,
    #     # T=4,
    # )

    print("Initializing model for inference")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, device_ids=[args.gpu])

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # Load model weights
    if args.resume:

        checkpoint = torch.load(args.resume, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise KeyError("Cannot find model weights in checkpoint")

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
        print('checkpoint is loaded')

    output_dir = '/home/qyb/frame16_feature'
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    if args.test_only:
        extract_features(model, data_loader_train, output_dir, device)
        # extract_features(model, data_loader_test, output_dir, device)
        print(f"features have saved in: {output_dir}")
        return

if __name__ == "__main__":
    args = parse_args()
    main(args)
