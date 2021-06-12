#!/usr/bin/env python3

import os
from glob import glob
import numpy as np

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch

import random
import pickle


class Loader(Dataset):
    def __init__(self, path):
        """

        """
        self.path = path
        self.data = pickle.load(open(self.path, 'rb'))
        self.images = self.data[0]
        self.masks = self.data[1]

    def shuffle(self, seed=42):
        random.Random(seed).shuffle(self.images)
        random.Random(seed).shuffle(self.masks)

        return self

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        im = self.images[idx] / 255
        msk = self.masks[idx]

        sample = {'image': im, 'mask': msk}

        return sample

    def collate_fn(self, samples):

        out = dict()
        image = [
            np.rollaxis(d['image'][..., None], -1).copy() for d in samples
        ]
        image = torch.stack([torch.from_numpy(im) for im in image]).float()
        out['image'] = image

        mask = [d['mask'].squeeze()[None, ...].copy() for d in samples]
        mask = torch.stack([torch.from_numpy(m) for m in mask]).float()
        out['mask'] = mask

        return out


if __name__ == "__main__":

    loader = Loader('../../data.dmp')

    for i, s in enumerate(loader):
        print('i: ', i)
        print('image shape: ', s['image'].shape)
