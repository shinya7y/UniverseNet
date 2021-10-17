import torch
from torch.nn import (GELU, BatchNorm2d, Conv2d, Identity, LayerNorm, Linear,
                      Module, ModuleList, ReLU, Sequential)

from .stochastic_depth import DropPath


class ConvStage(Module):

    def __init__(self,
                 num_blocks=2,
                 embedding_dim_in=64,
                 hidden_dim=128,
                 embedding_dim_out=128):
        super(ConvStage, self).__init__()
        self.conv_blocks = ModuleList()
        for i in range(num_blocks):
            block = Sequential(
                Conv2d(
                    embedding_dim_in,
                    hidden_dim,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False), BatchNorm2d(hidden_dim), ReLU(inplace=True),
                Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False), BatchNorm2d(hidden_dim), ReLU(inplace=True),
                Conv2d(
                    hidden_dim,
                    embedding_dim_in,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False), BatchNorm2d(embedding_dim_in),
                ReLU(inplace=True))
            self.conv_blocks.append(block)
        self.downsample = Conv2d(
            embedding_dim_in,
            embedding_dim_out,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1))

    def forward(self, x):
        for block in self.conv_blocks:
            x = x + block(x)
        return self.downsample(x)


class Mlp(Module):

    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=GELU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvMLPStage(Module):

    def __init__(self,
                 embedding_dim,
                 dim_feedforward=2048,
                 stochastic_depth_rate=0.1):
        super(ConvMLPStage, self).__init__()
        self.norm1 = LayerNorm(embedding_dim)
        self.channel_mlp1 = Mlp(
            embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.norm2 = LayerNorm(embedding_dim)
        self.connect = Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=embedding_dim,
            bias=False)
        self.connect_norm = LayerNorm(embedding_dim)
        self.channel_mlp2 = Mlp(
            embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.drop_path = DropPath(
            stochastic_depth_rate) if stochastic_depth_rate > 0 else Identity(
            )

    def forward(self, src):
        src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
        src = self.connect(self.connect_norm(src).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1)
        src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
        return src


class ConvDownsample(Module):

    def __init__(self, embedding_dim_in, embedding_dim_out):
        super().__init__()
        self.downsample = Conv2d(
            embedding_dim_in,
            embedding_dim_out,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        return x.permute(0, 2, 3, 1)


class BasicStage(Module):

    def __init__(self,
                 num_blocks,
                 embedding_dims,
                 mlp_ratio=1,
                 stochastic_depth_rate=0.1,
                 downsample=True):
        super(BasicStage, self).__init__()
        self.blocks = ModuleList()
        dpr = [
            x.item()
            for x in torch.linspace(0, stochastic_depth_rate, num_blocks)
        ]
        for i in range(num_blocks):
            block = ConvMLPStage(
                embedding_dim=embedding_dims[0],
                dim_feedforward=int(embedding_dims[0] * mlp_ratio),
                stochastic_depth_rate=dpr[i],
            )
            self.blocks.append(block)

        self.downsample_mlp = ConvDownsample(
            embedding_dims[0],
            embedding_dims[1]) if downsample else Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample_mlp(x)
        return x
