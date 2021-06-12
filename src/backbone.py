import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Any, List


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self,
                 inp: int,
                 oup: int,
                 stride: int,
                 ConvModule=nn.Conv2d) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp,
                                    inp,
                                    kernel_size=3,
                                    stride=self.stride,
                                    padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1),
            nn.BatchNorm2d(branch_features),
            ConvModule(branch_features,
                       branch_features,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int,
                       o: int,
                       kernel_size: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(i,
                         o,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias,
                         groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class Backbone(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        input_channels=3,
        num_classes: int = 1000,
        ConvModule=nn.Conv2d,
        inverted_residual: Callable[...,
                                    nn.Module] = InvertedResidual) -> None:
        super(Backbone, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        self._stage_out_channels = stages_out_channels

        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            ConvModule(input_channels,
                       output_channels,
                       kernel_size=3,
                       padding=2,
                       stride=1,
                       dilation=2,
                       bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [
                inverted_residual(input_channels, output_channels, 2,
                                  ConvModule)
            ]
            for i in range(repeats - 1):
                seq.append(
                    inverted_residual(output_channels, output_channels, 1,
                                      ConvModule))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        low_level_feats = x
        x = self.stage2(x)
        x = self.stage3(x)
        return x, low_level_feats
