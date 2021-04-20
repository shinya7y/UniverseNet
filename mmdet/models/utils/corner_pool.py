from mmcv.cnn import ConvModule
from mmcv.ops import CornerPool
from torch import nn


class CornerPoolPack(nn.Module):

    def __init__(self,
                 dim,
                 pool1,
                 pool2,
                 conv_cfg=None,
                 norm_cfg=None,
                 first_kernel_size=3,
                 kernel_size=3,
                 corner_dim=128):
        super(CornerPoolPack, self).__init__()
        self.p1_conv1 = ConvModule(
            dim,
            corner_dim,
            first_kernel_size,
            stride=1,
            padding=(first_kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.p2_conv1 = ConvModule(
            dim,
            corner_dim,
            first_kernel_size,
            stride=1,
            padding=(first_kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.p_conv1 = nn.Conv2d(corner_dim, dim, 3, padding=1, bias=False)
        self.p_gn1 = nn.GroupNorm(num_groups=32, num_channels=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ConvModule(
            dim,
            dim,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.pool1 = pool1
        self.pool2 = pool2

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_gn1 = self.p_gn1(p_conv1)

        conv1 = self.conv1(x)
        gn1 = self.gn1(conv1)
        relu1 = self.relu1(p_gn1 + gn1)

        conv2 = self.conv2(relu1)
        return conv2


class TLPool(CornerPoolPack):

    def __init__(self,
                 dim,
                 conv_cfg=None,
                 norm_cfg=None,
                 first_kernel_size=3,
                 kernel_size=3,
                 corner_dim=128):
        super(TLPool, self).__init__(
            dim,
            CornerPool('top'),
            CornerPool('left'),
            conv_cfg,
            norm_cfg,
            first_kernel_size,
            kernel_size,
            corner_dim,
        )


class BRPool(CornerPoolPack):

    def __init__(self,
                 dim,
                 conv_cfg=None,
                 norm_cfg=None,
                 first_kernel_size=3,
                 kernel_size=3,
                 corner_dim=128):
        super(BRPool, self).__init__(
            dim,
            CornerPool('bottom'),
            CornerPool('right'),
            conv_cfg,
            norm_cfg,
            first_kernel_size,
            kernel_size,
            corner_dim,
        )
