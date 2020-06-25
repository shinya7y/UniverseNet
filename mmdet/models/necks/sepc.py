import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import auto_fp16
from mmdet.ops.dcn.sepc_dconv import SEPCConv
# from mmdet.ops.dcn.sepc_dconv import ModulatedSEPCConv as SEPCConv
from ..builder import NECKS


@NECKS.register_module()
class SEPC(nn.Module):

    def __init__(self,
                 in_channels=[256] * 5,
                 out_channels=256,
                 num_outs=5,
                 stacked_convs=4,
                 pconv_deform=False,
                 lcconv_deform=False,
                 ibn=False,
                 lcconv_padding=0):
        super(SEPC, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert num_outs == 5
        self.fp16_enabled = False
        self.ibn = ibn
        self.pconvs = nn.ModuleList()

        for i in range(stacked_convs):
            self.pconvs.append(
                PConvModule(
                    in_channels[i],
                    out_channels,
                    ibn=self.ibn,
                    part_deform=pconv_deform))

        self.lconv = SEPCConv(
            256,
            256,
            kernel_size=3,
            padding=lcconv_padding,
            dilation=1,
            part_deform=lcconv_deform)
        self.cconv = SEPCConv(
            256,
            256,
            kernel_size=3,
            padding=lcconv_padding,
            dilation=1,
            part_deform=lcconv_deform)
        self.relu = nn.ReLU()
        if self.ibn:
            self.lbn = nn.BatchNorm2d(256)
            self.cbn = nn.BatchNorm2d(256)
        self.init_weights()

    def init_weights(self):
        for str in ["l", "c"]:
            m = getattr(self, str + "conv")
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        x = inputs
        for pconv in self.pconvs:
            x = pconv(x)
        cls_feats = [self.cconv(level, item) for level, item in enumerate(x)]
        loc_feats = [self.lconv(level, item) for level, item in enumerate(x)]
        if self.ibn:
            cls_feats = integrated_bn(cls_feats, self.cbn)
            loc_feats = integrated_bn(loc_feats, self.lbn)
        outs = [[self.relu(cls_feat), self.relu(loc_feat)]
                for cls_feat, loc_feat in zip(cls_feats, loc_feats)]
        return tuple(outs)


class PConvModule(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 kernel_size=[3, 3, 3],
                 dilation=[1, 1, 1],
                 groups=[1, 1, 1],
                 ibn=False,
                 part_deform=False):
        super(PConvModule, self).__init__()

        self.ibn = ibn
        self.pconv = nn.ModuleList()
        self.pconv.append(
            SEPCConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[0],
                dilation=dilation[0],
                groups=groups[0],
                padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2,
                part_deform=part_deform))
        self.pconv.append(
            SEPCConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[1],
                dilation=dilation[1],
                groups=groups[1],
                padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2,
                part_deform=part_deform))
        self.pconv.append(
            SEPCConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[2],
                dilation=dilation[2],
                groups=groups[2],
                padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2,
                stride=2,
                part_deform=part_deform))

        if self.ibn:
            self.bn = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.pconv:
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):
            temp_fea = self.pconv[1](level, feature)
            if level > 0:
                temp_fea += self.pconv[2](level, x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.interpolate(
                    self.pconv[0](level, x[level + 1]),
                    size=[temp_fea.size(2), temp_fea.size(3)],
                    mode='bilinear',
                    align_corners=True)
            next_x.append(temp_fea)
        if self.ibn:
            next_x = integrated_bn(next_x, self.bn)
        next_x = [self.relu(item) for item in next_x]
        return next_x


def integrated_bn(fms, bn):
    sizes = [p.shape[2:] for p in fms]
    n, c = fms[0].shape[0], fms[0].shape[1]
    print('integrated_bn a', flush=True)
    fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
    fm = bn(fm)
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
    print('integrated_bn b', flush=True)
    return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
