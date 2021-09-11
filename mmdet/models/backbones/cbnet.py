import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from .res2net import Res2Net
from .resnet import ResNet, _BatchNorm, build_norm_layer
from .swin_transformer import SwinTransformerOriginal

# TODO clarify the meaning of tmps


class CBSubResNetMixin:
    """Mixin class to modify ResNet for CBNetV2."""

    def _freeze_stages(self):
        """Freeze stages.

        We need to check the presence of layers possibly deleted by del_layers.
        """
        if self.frozen_stages >= 0:
            if self.deep_stem and hasattr(self, 'stem'):
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            elif hasattr(self, 'conv1'):
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if not hasattr(self, f'layer{i}'):
                continue
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def del_layers(self, del_stages):
        """Delete layers in lower stages."""
        self.del_stages = del_stages
        if self.del_stages >= 0:
            if self.deep_stem:
                del self.stem
            else:
                del self.conv1

        for i in range(1, self.del_stages + 1):
            delattr(self, f'layer{i}')

    def forward(self, x, cb_feats=None, pre_outs=None):
        """Forward function."""
        spatial_info = []
        outs = []

        if self.deep_stem and hasattr(self, 'stem'):
            x = self.stem(x)
            x = self.maxpool(x)
        elif hasattr(self, 'conv1'):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        else:
            x = pre_outs[0]
        outs.append(x)

        for i, layer_name in enumerate(self.res_layers):
            if hasattr(self, layer_name):
                res_layer = getattr(self, layer_name)
                spatial_info.append(x.shape[2:])
                if cb_feats is not None:
                    x = x + cb_feats[i]
                x = res_layer(x)
            else:
                x = pre_outs[i + 1]
            outs.append(x)
        return tuple(outs), spatial_info

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()


class CBSubResNet(CBSubResNetMixin, ResNet):
    """A ResNet backbone for CBNetV2."""

    def __init__(self, **kwargs):
        ResNet.__init__(self, **kwargs)


class CBSubRes2Net(CBSubResNetMixin, Res2Net):
    """A Res2Net backbone for CBNetV2."""

    def __init__(self, **kwargs):
        Res2Net.__init__(self, **kwargs)


class CBNetBase(BaseModule):
    """Base class for CBNetV2."""

    def _freeze_stages(self):
        """Freeze stages."""
        for m in self.cb_modules:
            m._freeze_stages()

    def init_cb_weights(self):
        """Initialize the weights in additional layers for CBNetV2."""
        raise NotImplementedError

    def init_weights(self):
        """Initialize the weights in backbone."""
        self.init_cb_weights()
        for m in self.cb_modules:
            m.init_weights()

    def _forward_cb_feats(self, feats, spatial_info):
        """Forward function to calculate CBNetV2 features."""
        raise NotImplementedError

    def forward(self, x):
        """Forward function."""
        outs_list = []
        cb_feats = None
        pre_outs = None
        for i, module in enumerate(self.cb_modules):
            pre_outs, spatial_info = module(x, cb_feats, pre_outs)
            outs = [pre_outs[j + 1] for j in self.out_indices]
            outs_list.append(tuple(outs))
            if i < len(self.cb_modules) - 1:
                cb_feats = self._forward_cb_feats(pre_outs, spatial_info)
        return tuple(outs_list)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        for m in self.cb_modules:
            m.train(mode=mode)
        self._freeze_stages()
        for m in self.cb_linears.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, _BatchNorm):
                m.eval()


class CBNet(CBNetBase):
    """CBNetV2 consisting of multiple backbones.

    Current implementation assumes that subnet is ResNet-like backbone.
    """

    def __init__(self,
                 subnet,
                 cb_inplanes,
                 cb_zero_init=True,
                 cb_del_stages=0,
                 cb_num_modules=2,
                 **kwargs):
        super(CBNet, self).__init__()
        self.cb_zero_init = cb_zero_init
        self.cb_del_stages = cb_del_stages
        self.cb_num_modules = cb_num_modules
        assert cb_num_modules >= 2

        self.cb_modules = nn.ModuleList()
        for cb_idx in range(self.cb_num_modules):
            cb_module = subnet(**kwargs)
            if cb_idx > 0:
                cb_module.del_layers(self.cb_del_stages)
            self.cb_modules.append(cb_module)
        self.out_indices = self.cb_modules[0].out_indices

        self.cb_linears = nn.ModuleList()
        self.num_layers = len(self.cb_modules[0].stage_blocks)
        norm_cfg = self.cb_modules[0].norm_cfg
        for i in range(self.num_layers):
            linears = nn.ModuleList()
            if i >= self.cb_del_stages:
                jrange = 4 - i
                for j in range(jrange):
                    linears.append(
                        nn.Sequential(
                            nn.Conv2d(
                                cb_inplanes[i + j + 1],
                                cb_inplanes[i],
                                1,
                                bias=False),
                            build_norm_layer(norm_cfg, cb_inplanes[i])[1]))

            self.cb_linears.append(linears)

    def init_cb_weights(self):
        """Initialize the weights in additional layers for CBNetV2."""
        if self.cb_zero_init:
            for ls in self.cb_linears:
                for m in ls:
                    if isinstance(m, nn.Sequential):
                        constant_init(m[-1], 0)
                    else:
                        constant_init(m, 0)

    def _forward_cb_feats(self, feats, spatial_info):
        """Forward function to calculate CBNetV2 features."""
        cb_feats = []
        for i in range(self.num_layers):
            if i >= self.cb_del_stages:
                h, w = spatial_info[i]
                feeds = []
                jrange = 4 - i
                for j in range(jrange):
                    tmp = self.cb_linears[i][j](feats[j + i + 1])
                    tmp = F.interpolate(tmp, size=(h, w), mode='nearest')
                    feeds.append(tmp)
                feed = torch.sum(torch.stack(feeds, dim=-1), dim=-1)
            else:
                feed = 0
            cb_feats.append(feed)

        return cb_feats


@BACKBONES.register_module()
class CBResNet(CBNet):
    """CBNetV2 consisting of multiple ResNets."""

    def __init__(self, **kwargs):
        super().__init__(subnet=CBSubResNet, **kwargs)


@BACKBONES.register_module()
class CBRes2Net(CBNet):
    """CBNetV2 consisting of multiple Res2Nets."""

    def __init__(self, **kwargs):
        super().__init__(subnet=CBSubRes2Net, **kwargs)


class CBSubSwinTransformer(SwinTransformerOriginal):
    """A Swin Transformer backbone for CBNetV2."""

    def _freeze_stages(self):
        """Freeze stages.

        We need to check the presence of layers possibly deleted by del_layers.
        """
        if self.frozen_stages >= 0 and hasattr(self, 'patch_embed'):
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                if m is None:
                    continue
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def del_layers(self, del_stages):
        """Delete layers in lower stages."""
        self.del_stages = del_stages
        if self.del_stages >= 0:
            del self.patch_embed

        if self.del_stages >= 1 and self.ape:
            del self.absolute_pos_embed

        for i in range(0, self.del_stages - 1):
            self.layers[i] = None

    def forward(self, x, cb_feats=None, pre_tmps=None):
        """Forward function."""
        outs = []
        tmps = []
        if hasattr(self, 'patch_embed'):
            x = self.patch_embed(x)
            Wh, Ww = x.size(2), x.size(3)
            if self.ape:
                # interpolate the position embedding to the corresponding size
                absolute_pos_embed = F.interpolate(
                    self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
                x = x + absolute_pos_embed
            x = x.flatten(2).transpose(1, 2)  # B Wh*Ww C
            x = self.pos_drop(x)
            tmps.append((x, Wh, Ww))
        else:
            x, Wh, Ww = pre_tmps[0]

        for i in range(self.num_layers):
            layer = self.layers[i]
            if layer is None:
                x_out, H, W, x, Wh, Ww = pre_tmps[i + 1]
            else:
                if cb_feats is not None:
                    x = x + cb_feats[i]
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            tmps.append((x_out, H, W, x, Wh, Ww))

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs), tmps

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(CBSubSwinTransformer, self).train(mode)
        self._freeze_stages()


@BACKBONES.register_module()
class CBSwinTransformer(BaseModule):
    """CBNetV2 consisting of multiple Swin Transformers."""

    def __init__(self,
                 embed_dim=96,
                 cb_zero_init=True,
                 cb_del_stages=1,
                 cb_num_modules=2,
                 **kwargs):
        super(CBSwinTransformer, self).__init__()
        self.cb_zero_init = cb_zero_init
        self.cb_del_stages = cb_del_stages
        self.cb_num_modules = cb_num_modules
        assert cb_num_modules >= 2

        self.cb_modules = nn.ModuleList()
        for cb_idx in range(self.cb_num_modules):
            cb_module = CBSubSwinTransformer(embed_dim=embed_dim, **kwargs)
            if cb_idx > 0:
                cb_module.del_layers(cb_del_stages)
            self.cb_modules.append(cb_module)

        self.num_layers = self.cb_modules[0].num_layers
        cb_inplanes = [embed_dim * 2**i for i in range(self.num_layers)]

        self.cb_linears = nn.ModuleList()
        for i in range(self.num_layers):
            linears = nn.ModuleList()
            if i >= self.cb_del_stages - 1:
                jrange = 4 - i
                for j in range(jrange):
                    if cb_inplanes[i + j] != cb_inplanes[i]:
                        layer = nn.Conv2d(cb_inplanes[i + j], cb_inplanes[i],
                                          1)
                    else:
                        layer = nn.Identity()
                    linears.append(layer)
            self.cb_linears.append(linears)

    def _freeze_stages(self):
        """Freeze stages."""
        for m in self.cb_modules:
            m._freeze_stages()

    def init_weights(self):
        """Initialize the weights in backbone."""
        # constant_init(self.cb_linears, 0)
        if self.cb_zero_init:
            for ls in self.cb_linears:
                for m in ls:
                    constant_init(m, 0)

        for m in self.cb_modules:
            m.init_weights()

    def spatial_interpolate(self, x, H, W):
        B, C = x.shape[:2]
        if H != x.shape[2] or W != x.shape[3]:
            # B, C, size[0], size[1]
            x = F.interpolate(x, size=(H, W), mode='nearest')
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # B, T, C
        return x

    def _forward_cb_feats(self, feats, tmps):
        """Forward function to calculate CBNetV2 features."""
        cb_feats = []
        Wh, Ww = tmps[0][-2:]
        for i in range(self.num_layers):
            feed = 0
            if i >= self.cb_del_stages - 1:
                jrange = 4 - i
                for j in range(jrange):
                    tmp = self.cb_linears[i][j](feats[j + i])
                    tmp = self.spatial_interpolate(tmp, Wh, Ww)
                    feed += tmp
            cb_feats.append(feed)
            Wh, Ww = tmps[i + 1][-2:]

        return cb_feats

    def forward(self, x):
        """Forward function."""
        outs = []
        cb_feats = None
        tmps = None
        for i, module in enumerate(self.cb_modules):
            feats, tmps = module(x, cb_feats, tmps)
            outs.append(feats)
            if i < len(self.cb_modules) - 1:
                cb_feats = self._forward_cb_feats(outs[-1], tmps)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(CBSwinTransformer, self).train(mode)
        for m in self.cb_modules:
            m.train(mode=mode)
        self._freeze_stages()
        for m in self.cb_linears.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, _BatchNorm):
                m.eval()
