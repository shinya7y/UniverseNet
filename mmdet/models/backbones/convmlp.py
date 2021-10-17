import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule, ModuleList, load_checkpoint

from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger


class ConvTokenizer(nn.Module):
    """Convolutional Tokenizer (deep_stem) of ConvMLP."""

    def __init__(self, in_dim=3, embed_dim=64):
        super(ConvTokenizer, self).__init__()
        hidden_dim = embed_dim // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class ConvStage(nn.Module):
    """Convolution Stage of ConvMLP."""

    def __init__(self,
                 num_blocks=2,
                 embed_dim_in=64,
                 hidden_dim=128,
                 embed_dim_out=128):
        super(ConvStage, self).__init__()
        self.conv_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2d(embed_dim_in, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, embed_dim_in, 1, bias=False),
                nn.BatchNorm2d(embed_dim_in), nn.ReLU(inplace=True))
            self.conv_blocks.append(block)
        self.downsample = nn.Conv2d(
            embed_dim_in, embed_dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        """Forward function."""
        for block in self.conv_blocks:
            x = x + block(x)
        return self.downsample(x)


class Mlp(nn.Module):
    """Multilayer perceptron used in ConvMLP."""

    def __init__(self,
                 embed_dim_in,
                 hidden_dim=None,
                 embed_dim_out=None,
                 activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim_in
        embed_dim_out = embed_dim_out or embed_dim_in
        self.fc1 = nn.Linear(embed_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, embed_dim_out)

    def forward(self, x):
        """Forward function."""
        return self.fc2(self.act(self.fc1(x)))


class ConvMLPBlock(nn.Module):
    """Conv-MLP block of ConvMLP."""

    def __init__(self, embed_dim, dim_feedforward=2048, drop_path_rate=0.1):
        super(ConvMLPBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.channel_mlp1 = Mlp(
            embed_dim_in=embed_dim, hidden_dim=dim_feedforward)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.connect = nn.Conv2d(
            embed_dim,
            embed_dim,
            3,
            stride=1,
            padding=1,
            groups=embed_dim,
            bias=False)
        self.connect_norm = nn.LayerNorm(embed_dim)
        self.channel_mlp2 = Mlp(
            embed_dim_in=embed_dim, hidden_dim=dim_feedforward)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, src):
        """Forward function."""
        src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
        src = self.connect_norm(src).permute(0, 3, 1, 2)
        src = self.connect(src).permute(0, 2, 3, 1)
        src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
        return src


class ConvDownsample(nn.Module):
    """Convolutional Downsampling of ConvMLP."""

    def __init__(self, embed_dim_in, embed_dim_out):
        super().__init__()
        self.downsample = nn.Conv2d(
            embed_dim_in, embed_dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        """Forward function."""
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        return x.permute(0, 2, 3, 1)


class ConvMLPStage(nn.Module):
    """Conv-MLP Stage of ConvMLP."""

    def __init__(self,
                 num_blocks,
                 embed_dims,
                 mlp_ratio=1,
                 drop_path_rate=0.1,
                 downsample=True):
        super(ConvMLPStage, self).__init__()
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        for i in range(num_blocks):
            block = ConvMLPBlock(
                embed_dim=embed_dims[0],
                dim_feedforward=int(embed_dims[0] * mlp_ratio),
                drop_path_rate=dpr[i])
            self.blocks.append(block)
        if downsample:
            self.downsample_mlp = ConvDownsample(embed_dims[0], embed_dims[1])
        else:
            self.downsample_mlp = nn.Identity()

    def forward(self, x):
        """Forward function."""
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample_mlp(x)
        return x


@BACKBONES.register_module()
class ConvMLP(BaseModule):
    """ConvMLP backbone.

    https://arxiv.org/abs/2109.04454
    """

    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 in_channels=3,
                 stem_channels=64,
                 num_conv_blocks=3,
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if out_indices != (0, 1, 2, 3):
            raise NotImplementedError
        assert len(blocks) == len(dims) == len(mlp_ratios), \
            'blocks, dims and mlp_ratios must agree in size, ' \
            f'{len(blocks)}, {len(dims)} and {len(mlp_ratios)} passed.'

        self.tokenizer = ConvTokenizer(
            in_dim=in_channels, embed_dim=stem_channels)
        self.conv_stages = ConvStage(
            num_conv_blocks,
            embed_dim_in=stem_channels,
            hidden_dim=dims[0],
            embed_dim_out=dims[0])
        self.stages = ModuleList()
        for i in range(0, len(blocks)):
            is_last_stage = i == len(blocks) - 1
            stage = ConvMLPStage(
                num_blocks=blocks[i],
                embed_dims=dims[i:i + 2],
                mlp_ratio=mlp_ratios[i],
                drop_path_rate=0.1,
                downsample=(not is_last_stage))
            self.stages.append(stage)

    def forward(self, x):
        """Forward function."""
        outs = []
        x = self.tokenizer(x)
        outs.append(x)  # feature map F1
        x = self.conv_stages(x)
        outs.append(x)  # feature map F2
        x = x.permute(0, 2, 3, 1)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # skip second last stage whose resolution is the same as last stage
            if i == len(self.stages) - 2:
                continue
            outs.append(x.permute(0, 3, 1, 2).contiguous())  # feat map F3, F4
        return tuple(outs)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def init_weights(self):
        """Initialize the weights in backbone."""
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if not isinstance(self.init_cfg.checkpoint, str):
                raise TypeError('init_cfg.checkpoint must be str')
            load_checkpoint(
                self,
                self.init_cfg.checkpoint,
                logger=logger,
                map_location='cpu')


@BACKBONES.register_module()
class ConvMLPSmall(ConvMLP):
    """ConvMLP-S."""

    def __init__(self, **kwargs):
        super(ConvMLPSmall, self).__init__(
            blocks=[2, 4, 2],
            dims=[128, 256, 512],
            mlp_ratios=[2, 2, 2],
            stem_channels=64,
            num_conv_blocks=2,
            **kwargs)


@BACKBONES.register_module()
class ConvMLPMedium(ConvMLP):
    """ConvMLP-M."""

    def __init__(self, **kwargs):
        super(ConvMLPMedium, self).__init__(
            blocks=[3, 6, 3],
            dims=[128, 256, 512],
            mlp_ratios=[3, 3, 3],
            stem_channels=64,
            num_conv_blocks=3,
            **kwargs)


@BACKBONES.register_module()
class ConvMLPLarge(ConvMLP):
    """ConvMLP-L."""

    def __init__(self, **kwargs):
        super(ConvMLPLarge, self).__init__(
            blocks=[4, 8, 3],
            dims=[192, 384, 768],
            mlp_ratios=[3, 3, 3],
            stem_channels=96,
            num_conv_blocks=3,
            **kwargs)
