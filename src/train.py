#!/usr/bin/env python3

import os

import click
import matplotlib.pyplot as plt
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

import utils as utls
from defaults import default_args as args
from loader import Loader
from looper import Looper
from loss import DiceLoss
from network import MySegmentationModel


@click.command()
@click.option('-i',
              '--in_path',
              type=str,
              required=True,
              help='Path to input data.')
@click.option('-o',
              '--out_path',
              type=str,
              required=True,
              help='Path to output')
@click.option('--epochs',
              type=int,
              default=args['epochs'],
              help='Number of epochs to train')
@click.option('--train_split',
              type=float,
              default=args['train_split'],
              help='Ratio of images to use for training')
@click.option('-bs',
              '--batch_size',
              type=int,
              default=args['batch_size'],
              help='Batch size')
@click.option('-lrm',
              '--lr_milestones',
              type=list,
              default=args['lr_milestones'],
              help='Milestones to decrease learning rate')
@click.option('-lg',
              '--lr_gamma',
              type=float,
              default=args['lr_gamma'],
              help='Decrease factor to apply to learning rate')
@click.option('-c', '--cuda', is_flag=True, help='Use CUDA')
@click.option('--bce',
              is_flag=True,
              help='Use BCE loss instead of Dice loss (default)')
@click.option('--coordconv', is_flag=True, help='Use CoordConv')
@click.option('--decay',
              type=float,
              default=args['decay'],
              help='Weight decay')
@click.option('--pos-weight',
              type=float,
              default=args['coordconv'],
              help='Weights applied to classes in BCE')
@click.option('--smooth',
              type=float,
              default=args['smooth'],
              help='Smoothing factor applied to DICE loss')
@click.option('--momentum',
              type=float,
              default=args['momentum'],
              help='Momentum of SGD')
@click.option('--cp-pred',
              type=int,
              default=args['cp_pred'],
              help='Save model period')
@click.option('-lr',
              '--learning_rate',
              default=args['learning_rate'],
              help='Initial learning rate (lr_scheduler is applied).')
@click.option('--adam', is_flag=True, help='Use Adam optimizer')
def train_cmd(in_path: str, out_path: str, epochs: int, train_split: float,
              batch_size: int, learning_rate: float, lr_milestones: list,
              lr_gamma: float, cp_pred: int, decay: float, momentum: float,
              cuda: bool, pos_weight: list, smooth: float, coordconv: bool,
              bce: bool, adam: bool):

    train(in_path, out_path, epochs, train_split, batch_size, learning_rate,
          lr_milestones, lr_gamma, cp_pred, decay, momentum, cuda, pos_weight,
          smooth, coordconv, bce, adam)


def train(in_path: str, out_path: str, epochs: int, train_split: float,
          batch_size: int, learning_rate: float, lr_milestones: list,
          lr_gamma: float, cp_pred: int, decay: float, momentum: float,
          cuda: bool, pos_weight: list, smooth: float, coordconv: bool,
          bce: bool, adam: bool):
    torch.backends.cudnn.benchmark = True

    # create output dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # save arguments to file
    args = locals()
    with open(os.path.join(out_path, 'params.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

    # create dataloader and split it into train, val, and test sets
    batch_size = batch_size
    dset = Loader(in_path).shuffle(seed=42)
    dls = utls.do_splits(dset, train_split, (1 - train_split) / 2,
                         (1 - train_split) / 2)
    dls = {
        'train':
        DataLoader(dls['train'],
                   collate_fn=dset.collate_fn,
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=batch_size),
        'val':
        DataLoader(dls['val'],
                   collate_fn=dset.collate_fn,
                   batch_size=batch_size,
                   num_workers=batch_size)
    }

    dls['train'].shuffle = True

    device = torch.device('cuda' if cuda else 'cpu')
    model = MySegmentationModel(coord_conv=coordconv).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('num. of parameters: ', params)

    # create directories
    dirs = {
        'run': out_path,
        'prev': os.path.join(out_path, 'prevs'),
        'cps': os.path.join(out_path, 'cps'),
    }
    for d in dirs.values():
        if not os.path.exists(d):
            os.makedirs(d)

    writer = SummaryWriter(dirs['run'], flush_secs=1)

    criterions = {
        k: torch.nn.BCEWithLogitsLoss() if bce else DiceLoss(
            smooth=smooth if k == 'train' else 0)
        for k in ['train', 'val']
    }

    if adam:
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=decay)

    lr_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=lr_milestones,
                                                  gamma=lr_gamma)

    loopers = {
        key: Looper(model, dls[key], key, optimizer, criterions[key], writer,
                    device, lr_sch if key == 'train' else None)
        for key in ['train', 'val']
    }

    for epoch in range(epochs):

        for phase in ['train', 'val']:
            loopers[phase].loop(epoch, epochs)

            # save plots
            if phase == 'val':
                fig = loopers['val'].make_plot()
                fig.savefig(os.path.join(dirs['prev'],
                                         'ep_{:03d}.png'.format(epoch + 1)),
                            bbox_inches='tight')
                plt.close()

            if ((epoch + 1) % cp_pred == 0) & (epoch > 0) & (phase == 'train'):
                print('saving checkpoint')
                utls.save_cp(
                    model,
                    os.path.join(dirs['cps'],
                                 'ep_{:04d}.pth.tar'.format(epoch)))


if __name__ == '__main__':
    train_cmd()
