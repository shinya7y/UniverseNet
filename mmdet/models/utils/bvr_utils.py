import types
import warnings

from torch import nn
from torch.nn import functional as F

from mmdet.core import multi_apply
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.dense_heads.retina_head import RetinaHead


def anchorfree_forward_features(self, feats):
    return multi_apply(self.forward_feature_single, feats)


def anchorfree_forward_feature_single(self, x):
    cls_feat = x
    reg_feat = x
    for cls_layer in self.cls_convs:
        cls_feat = cls_layer(cls_feat)
    for reg_layer in self.reg_convs:
        reg_feat = reg_layer(reg_feat)
    return cls_feat, reg_feat


def atss_forward_predictions(self, cls_feats, reg_feats):
    return multi_apply(self.forward_prediction_single, cls_feats, reg_feats,
                       self.scales)


def atss_forward_prediction_single(self, cls_feat, reg_feat, scale):
    cls_score = self.atss_cls(cls_feat)
    bbox_pred = scale(self.atss_reg(reg_feat)).float()
    centerness = self.atss_centerness(reg_feat)
    return cls_score, bbox_pred, centerness


def fcos_forward_predictions(self, cls_feats, reg_feats):
    return multi_apply(self.forward_prediction_single, cls_feats, reg_feats,
                       self.scales, self.strides)


def fcos_forward_prediction_single(self, cls_feat, reg_feat, scale, stride):
    cls_score = self.conv_cls(cls_feat)
    bbox_pred = self.conv_reg(reg_feat)
    if self.centerness_on_reg:
        centerness = self.conv_centerness(reg_feat)
    else:
        centerness = self.conv_centerness(cls_feat)
    # scale the bbox_pred of different level
    # float to avoid overflow when enabling FP16
    bbox_pred = scale(bbox_pred).float()
    if self.norm_on_bbox:
        bbox_pred = F.relu(bbox_pred)
        if not self.training:
            bbox_pred *= stride
    else:
        bbox_pred = bbox_pred.exp()
    return cls_score, bbox_pred, centerness


def retina_forward_predictions(self, cls_feats, reg_feats):
    return multi_apply(self.forward_prediction_single, cls_feats, reg_feats)


def retina_forward_prediction_single(self, cls_feat, reg_feat):
    cls_score = self.retina_cls(cls_feat)
    bbox_pred = self.retina_reg(reg_feat)
    return cls_score, bbox_pred


def assign_methods_for_bvr(module: nn.Module):
    """Modify the structure of bbox_head by assigning methods.

    Args:
        module: bbox_head
    """
    # check whether the bbox_head already has required methods
    if hasattr(module, 'forward_features') and hasattr(module,
                                                       'forward_predictions'):
        return

    # check whether BVR supports the bbox_head
    supported_heads = ('ATSSHead', 'FCOSHead', 'RetinaHead')
    module_name = module.__class__.__name__
    assert module_name in supported_heads, 'not supported bbox_head'
    assert hasattr(module, 'cls_convs'), 'not found cls_convs'
    assert hasattr(module, 'reg_convs'), 'not found reg_convs'

    # warning
    warnings.warn(f'Methods for BVR will be assigned to {module_name}.'
                  ' The bbox_head may break if the prediction'
                  ' is maintained by other branches.')

    # assign forward_features, forward_feature_single
    module.forward_features = types.MethodType(anchorfree_forward_features,
                                               module)
    module.forward_feature_single = types.MethodType(
        anchorfree_forward_feature_single, module)

    # assign forward_predictions, forward_prediction_single
    if isinstance(module, ATSSHead):
        module.forward_predictions = types.MethodType(atss_forward_predictions,
                                                      module)
        module.forward_prediction_single = types.MethodType(
            atss_forward_prediction_single, module)
    elif isinstance(module, FCOSHead):
        module.forward_predictions = types.MethodType(fcos_forward_predictions,
                                                      module)
        module.forward_prediction_single = types.MethodType(
            fcos_forward_prediction_single, module)
    elif isinstance(module, RetinaHead):
        module.forward_predictions = types.MethodType(
            retina_forward_predictions, module)
        module.forward_prediction_single = types.MethodType(
            retina_forward_prediction_single, module)
    else:
        assert False, 'this line should be unreachable'

    return
