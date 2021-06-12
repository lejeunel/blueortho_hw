#!/usr/bin/env python3

import torch
from torch import nn
from backbone import Backbone
from deeplab import DeepLabHeadV3Plus
from coordconv import CoordConv


class MySegmentationModel(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 stages_repeats=[2, 4, 2],
                 stages_out_channels=[8, 16, 32],
                 coord_conv=False):

        super(MySegmentationModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        ConvModule = CoordConv if coord_conv else nn.Conv2d
        self.backbone = Backbone(input_channels=1,
                                 stages_repeats=stages_repeats,
                                 stages_out_channels=stages_out_channels,
                                 ConvModule=ConvModule)

        self.head = DeepLabHeadV3Plus(
            aspp_channels=stages_out_channels[-1],
            low_level_channels=stages_out_channels[0],
            num_classes=1)

    def forward(self, image):

        high_level_feats, low_level_feats = self.backbone(image)
        out = self.head({
            'low_level': low_level_feats,
            'out': high_level_feats
        })

        return out


if __name__ == "__main__":

    from loader import Loader
    import utils as utls
    from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler
    model = MySegmentationModel()
    dset = Loader('../../data.dmp')
    loader = DataLoader(dset,
                        collate_fn=dset.collate_fn,
                        batch_size=4,
                        shuffle=True)

    for i, s in enumerate(loader):
        s = utls.batch_to_device(s, 'cpu')
        model(s['image'])
