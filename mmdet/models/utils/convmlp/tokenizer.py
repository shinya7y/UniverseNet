import torch.nn as nn


class ConvTokenizer(nn.Module):

    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                3,
                embedding_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False), nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                embedding_dim // 2,
                embedding_dim // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False), nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                embedding_dim // 2,
                embedding_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False), nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1)))

    def forward(self, x):
        return self.block(x)
