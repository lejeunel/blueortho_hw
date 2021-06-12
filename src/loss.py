#!/usr/bin/env python3

from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = inputs.sigmoid()

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection +
                self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice
