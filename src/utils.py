#!/usr/bin/env python3

import torch
from torch.utils.data import random_split
import os


def save_cp(model, path):
    dir_ = os.path.split(path)[0]
    if (not os.path.exists(dir_)):
        os.makedirs(dir_)

    state_dict = model.state_dict()

    torch.save(state_dict, path)


def do_splits(dl, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):

    assert train_ratio + val_ratio + test_ratio == 1., 'ratios must sum to 1'

    train_size = int(len(dl) * train_ratio)
    val_test_size = len(dl) - train_size
    val_size = int(val_test_size * val_ratio)
    test_size = len(dl) - train_size - val_size

    splits = random_split(dl, [train_size, val_size, test_size],
                          generator=torch.Generator().manual_seed(seed))

    return {'train': splits[0], 'val': splits[1], 'test': splits[2]}


def batch_to_device(batch, device):

    return {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }
