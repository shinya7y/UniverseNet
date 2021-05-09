import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.deform_conv import DeformConv2d, deform_conv2d
from mmcv.ops.modulated_deform_conv import (ModulatedDeformConv2d,
                                            modulated_deform_conv2d)
from torch.nn.modules.utils import _pair

# from mmdet.ops.dcn import (DeformConv2d, ModulatedDeformConv2d,
#                            deform_conv2d, modulated_deform_conv2d)


class SEPCConv(DeformConv2d):
    """DCNv1-based scale-equalizing module of SEPC."""

    def __init__(self, *args, part_deform=False, **kwargs):
        super(SEPCConv, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        if self.part_deform:
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 2 * self.kernel_size[0] *
                self.kernel_size[1],
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                dilation=_pair(self.dilation),
                bias=True)
            self.init_offset()

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1

    def init_offset(self):
        """Initialize the weights of conv_offset for DCN."""
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        """Forward function."""
        if i < self.start_level or not self.part_deform:
            return torch.nn.functional.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups)

        offset = self.conv_offset(x)

        # padding is needed to avoid error `input image is smaller than kernel`
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) <
                                                          self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0)
            offset = offset.contiguous()

        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        bias = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type_as(out)
        return out + bias


class ModulatedSEPCConv(ModulatedDeformConv2d):
    """DCNv2-based scale-equalizing module of SEPC."""

    _version = 2

    def __init__(self, *args, part_deform=False, **kwargs):
        super(ModulatedSEPCConv, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        if self.part_deform:
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 3 * self.kernel_size[0] *
                self.kernel_size[1],
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                dilation=_pair(self.dilation),
                bias=True)
            self.init_offset()

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1

    def init_offset(self):
        """Initialize the weights of conv_offset for DCN."""
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        """Forward function."""
        if i < self.start_level or not self.part_deform:
            return torch.nn.functional.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups)

        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(
            x, offset, mask, self.weight, None, self.stride, self.padding,
            self.dilation, self.groups, self.deform_groups
        ) + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
