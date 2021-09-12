# Copyright (c) OpenMMLab. All rights reserved.
from .cbnet import CBRes2Net, CBResNet, CBSwinTransformer
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import pvt_large, pvt_medium, pvt_small, pvt_tiny
from .pvt_v2 import (pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_li, pvt_v2_b3,
                     pvt_v2_b4, pvt_v2_b5)
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .swin_transformer import SwinTransformerOriginal
from .trident_resnet import TridentResNet

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer'
]

__all__ += [
    'SwinTransformerOriginal', 'CBResNet', 'CBRes2Net', 'CBSwinTransformer'
]
__all__ += [
    'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large', 'pvt_v2_b0',
    'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b2_li', 'pvt_v2_b3', 'pvt_v2_b4',
    'pvt_v2_b5'
]
