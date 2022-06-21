# Copyright (c) OpenMMLab. All rights reserved.
from .cbnet import CBRes2Net, CBResNet, CBSwinTransformer
from .convmlp import ConvMLP, ConvMLPLarge, ConvMLPMedium, ConvMLPSmall
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .focalnet import FocalNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .poolformer import (poolformer_m36_feat, poolformer_m48_feat,
                         poolformer_s12_feat, poolformer_s24_feat,
                         poolformer_s36_feat)
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .pvtv2_original import (pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_li,
                             pvt_v2_b3, pvt_v2_b4, pvt_v2_b5)
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .swin_transformer import SwinTransformerOriginal
from .swin_transformer_v2 import SwinTransformerV2
from .trident_resnet import TridentResNet

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet'
]

__all__ += [
    'SwinTransformerOriginal', 'CBResNet', 'CBRes2Net', 'CBSwinTransformer',
    'ConvMLP', 'ConvMLPLarge', 'ConvMLPMedium', 'ConvMLPSmall', 'FocalNet',
    'SwinTransformerV2'
]
__all__ += [
    'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b2_li', 'pvt_v2_b3',
    'pvt_v2_b4', 'pvt_v2_b5'
]
__all__ += [
    'poolformer_s12_feat', 'poolformer_s24_feat', 'poolformer_s36_feat',
    'poolformer_m36_feat', 'poolformer_m48_feat'
]
