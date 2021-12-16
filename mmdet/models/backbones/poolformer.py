# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple

from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger


class PatchEmbed(nn.Module):
    """Patch Embedding implemented by a layer of conv.

    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """Forward function."""
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """LayerNorm only for channel dimension."""

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Forward function."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """Group Normalization with 1 group."""

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """Implementation of pooling for PoolFormer."""

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False)

    def forward(self, x):
        """Forward function."""
        return self.pool(x) - x


class Mlp(nn.Module):
    """Implementation of MLP with 1*1 convolutions."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_rate)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward function."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """Implementation of one PoolFormer block.

    Stochastic depth and LayerScale are useful to train deep PoolFormers.

    Args:
        dim: embedding dim
        pool_size: pooling size
        mlp_ratio: mlp expansion ratio
        act_layer: activation
        norm_layer: normalization
        drop_rate: dropout rate
        drop_path_rate: Stochastic depth rate,
            refer to https://arxiv.org/abs/1603.09382
        use_layer_scale: whether to use LayerScale,
            refer to https://arxiv.org/abs/2103.17239
        layer_scale_init_value: LayerScale initial value
    """

    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=GroupNorm,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_rate=drop_rate)

        if drop_path_rate > 0.:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """Forward function."""
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
                self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
                self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim,
                 index,
                 layers,
                 pool_size=3,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=GroupNorm,
                 drop_rate=.0,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):
    """Generate PoolFormer blocks for a stage."""
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (
            sum(layers) - 1)
        blocks.append(
            PoolFormerBlock(
                dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value))
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(BaseModule):
    """PoolFormer backbone.

    Args:
        layers: number of blocks for the 4 stages.
        embed_dims: embedding dims for the 4 stages.
        mlp_ratios: mlp ratios for the 4 stages.
        downsamples: flags to apply downsampling or not.
        pool_size: pooling size for the 4 stages.
        norm_layer: define the types of normalization.
        act_layer: define the types of activation.
        in_patch_size: specify the patch embedding for the input image.
        in_stride: specify the patch embedding for the input image.
        in_pad: specify the patch embedding for the input image.
        down_patch_size: specify the downsample (patch embedding).
        down_stride: specify the downsample (patch embedding).
        down_pad: specify the downsample (patch embedding).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 layers,
                 embed_dims=None,
                 mlp_ratios=None,
                 downsamples=None,
                 pool_size=3,
                 norm_layer=GroupNorm,
                 act_layer=nn.GELU,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 out_indices=(0, 2, 4, 6),
                 init_cfg=None):
        self.out_indices = out_indices
        assert len(layers) == len(embed_dims) == len(mlp_ratios) == len(
            downsamples)
        super().__init__(init_cfg=init_cfg)

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        embed_dims_including_down = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            embed_dims_including_down.append(embed_dims[i])
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1]))
                embed_dims_including_down.append(embed_dims[i + 1])
        self.network = ModuleList(network)

        # add a norm layer for each output
        for idx, embed_dim in enumerate(embed_dims_including_down):
            if idx in self.out_indices:
                layer = norm_layer(embed_dim)
                layer_name = f'norm{idx}'
                self.add_module(layer_name, layer)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        """Initialize the weights in backbone."""
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for '
                           f'{self.__class__.__name__}, '
                           f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            logger.warning(f'missing_keys: {missing_keys}')
            logger.warning(f'unexpected_keys: {unexpected_keys}')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs


@BACKBONES.register_module()
class poolformer_s12_feat(PoolFormer):
    """PoolFormer-S12 model, Params: 12M."""

    def __init__(self, **kwargs):
        super().__init__(
            layers=(2, 2, 6, 2),
            embed_dims=(64, 128, 320, 512),
            mlp_ratios=(4, 4, 4, 4),
            downsamples=(True, True, True, True),
            **kwargs)


@BACKBONES.register_module()
class poolformer_s24_feat(PoolFormer):
    """PoolFormer-S24 model, Params: 21M."""

    def __init__(self, **kwargs):
        super().__init__(
            layers=(4, 4, 12, 4),
            embed_dims=(64, 128, 320, 512),
            mlp_ratios=(4, 4, 4, 4),
            downsamples=(True, True, True, True),
            **kwargs)


@BACKBONES.register_module()
class poolformer_s36_feat(PoolFormer):
    """PoolFormer-S36 model, Params: 31M."""

    def __init__(self, **kwargs):
        super().__init__(
            layers=(6, 6, 18, 6),
            embed_dims=(64, 128, 320, 512),
            mlp_ratios=(4, 4, 4, 4),
            downsamples=(True, True, True, True),
            layer_scale_init_value=1e-6,
            **kwargs)


@BACKBONES.register_module()
class poolformer_m36_feat(PoolFormer):
    """PoolFormer-M36 model, Params: 56M."""

    def __init__(self, **kwargs):
        super().__init__(
            layers=(6, 6, 18, 6),
            embed_dims=(96, 192, 384, 768),
            mlp_ratios=(4, 4, 4, 4),
            downsamples=(True, True, True, True),
            layer_scale_init_value=1e-6,
            **kwargs)


@BACKBONES.register_module()
class poolformer_m48_feat(PoolFormer):
    """PoolFormer-M48 model, Params: 73M."""

    def __init__(self, **kwargs):
        super().__init__(
            layers=(8, 8, 24, 8),
            embed_dims=(96, 192, 384, 768),
            mlp_ratios=(4, 4, 4, 4),
            downsamples=(True, True, True, True),
            layer_scale_init_value=1e-6,
            **kwargs)
