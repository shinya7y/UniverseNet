import math
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, load_checkpoint
from mmcv.utils import to_2tuple

from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger


class DWConv(nn.Module):
    """Depth-wise convolution with reshape for PVTv2."""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        """Forward function."""
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    """Multilayer perceptron used in PVTv2."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop_rate=0.,
                 linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        """Forward function."""
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Spatial-Reduction Attention (SRA) of PVTv2."""

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, \
            f'dim {dim} should be divisible by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.sr_ratio = sr_ratio
        self.linear = linear
        if linear:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        else:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(
                    dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        """Forward function."""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.linear:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
        else:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
            else:
                x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """PVTv2 Block."""

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            sr_ratio=sr_ratio,
            linear=linear)
        # NOTE: drop path for stochastic depth,
        # we shall see if this is better than dropout here
        if drop_path_rate > 0.:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_rate=drop_rate,
            linear=linear)

    def forward(self, x, H, W):
        """Forward function."""
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Forward function."""
        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2Original(BaseModule):
    """Pyramid Vision Transformer v2 backbone.

    The original implementation of PVTv2 with minor modifications. Please
    consider using the mmdet's implementation in pvt.py when you train new
    models.
    """

    def __init__(self,
                 patch_sizes=(7, 3, 3, 3),
                 strides=(4, 2, 2, 2),
                 in_channels=3,
                 embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3),
                 sr_ratios=(8, 4, 2, 1),
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 linear=False,
                 pretrained=None,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.depths = depths
        self.num_stages = num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.linear = linear
        self.pretrained = pretrained

        # stochastic depth decay rule
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        depth_cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=patch_sizes[i],
                stride=strides[i],
                in_channels=in_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[depth_cur + depth_idx],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    linear=linear) for depth_idx in range(depths[i])
            ])
            norm = norm_layer(embed_dims[i])
            depth_cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        """Initialize the weights in backbone."""
        self.apply(self._init_weights)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self,
                self.pretrained,
                map_location='cpu',
                strict=False,
                logger=logger)
        elif self.pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def freeze_patch_emb(self):
        """Freeze the first patch_embed."""
        self.patch_embed1.requires_grad = False

    def forward(self, x):
        """Forward function."""
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)

        return outs


@BACKBONES.register_module()
class pvt_v2_b0(PyramidVisionTransformerV2Original):
    """PVTv2-B0."""

    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            embed_dims=(32, 64, 160, 256),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(8, 8, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(2, 2, 2, 2),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


@BACKBONES.register_module()
class pvt_v2_b1(PyramidVisionTransformerV2Original):
    """PVTv2-B1."""

    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            embed_dims=(64, 128, 320, 512),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(8, 8, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(2, 2, 2, 2),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


@BACKBONES.register_module()
class pvt_v2_b2(PyramidVisionTransformerV2Original):
    """PVTv2-B2."""

    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            embed_dims=(64, 128, 320, 512),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(8, 8, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(3, 4, 6, 3),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


@BACKBONES.register_module()
class pvt_v2_b2_li(PyramidVisionTransformerV2Original):
    """PVTv2-B2-Li."""

    def __init__(self, **kwargs):
        super(pvt_v2_b2_li, self).__init__(
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            embed_dims=(64, 128, 320, 512),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(8, 8, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(3, 4, 6, 3),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1,
            linear=True,
            **kwargs)


@BACKBONES.register_module()
class pvt_v2_b3(PyramidVisionTransformerV2Original):
    """PVTv2-B3."""

    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            embed_dims=(64, 128, 320, 512),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(8, 8, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(3, 4, 18, 3),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


@BACKBONES.register_module()
class pvt_v2_b4(PyramidVisionTransformerV2Original):
    """PVTv2-B4."""

    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            embed_dims=(64, 128, 320, 512),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(8, 8, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(3, 8, 27, 3),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


@BACKBONES.register_module()
class pvt_v2_b5(PyramidVisionTransformerV2Original):
    """PVTv2-B5."""

    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            embed_dims=(64, 128, 320, 512),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(4, 4, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(3, 6, 40, 3),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)
