import torch
from torch import nn
from torch.nn import functional as F


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self,
                 aspp_channels,
                 low_level_channels,
                 num_classes,
                 proj_channels=16,
                 out_channels=32,
                 aspp_dilate=[2, 4],
                 ConvModule=nn.Conv2d):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            ConvModule(low_level_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(aspp_channels,
                         aspp_dilate,
                         out_channels=proj_channels * 3,
                         ConvModule=ConvModule)

        self.classifier = nn.Sequential(
            ConvModule(4 * proj_channels,
                       out_channels,
                       3,
                       padding=1,
                       bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), ConvModule(out_channels, num_classes, 1))
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature,
                                       size=low_level_feature.shape[2:],
                                       mode='bilinear',
                                       align_corners=False)
        return self.classifier(
            torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 ConvModule=nn.Conv2d):
        modules = [
            ConvModule(in_channels,
                       out_channels,
                       3,
                       padding=dilation,
                       dilation=dilation,
                       bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, ConvModule=nn.Conv2d):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x,
                             size=size,
                             mode='bilinear',
                             align_corners=False)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 atrous_rates,
                 out_channels=256,
                 ConvModule=nn.Conv2d):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(ConvModule(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))

        rate1, rate2 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            ConvModule(len(modules) * out_channels,
                       out_channels,
                       1,
                       bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride, module.padding,
                                                module.dilation, module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
