#!/usr/bin/env python3

# set default arguments
default_args = {
    'epochs': 20,
    'train_split': 0.6,
    'batch_size': 12,
    'lr_milestones': [15],
    'lr_gamma': 0.1,
    'decay': 0,
    'pos_weight': 1.,
    'cp_pred': 10,
    'momentum': 0.1,
    'smooth': 0.1,
    'learning_rate': 1e-3,
    'coordconv': False,
    'adam': False,
    'cuda': False,
    'bce': False
}
