import warnings
from typing import Tuple

import torch
from mmcv.cnn import ConvModule, normal_init
from torch import nn
from torch.nn import functional as F

from .builder import BVR_POSITIONAL_ENCODING


class BasePositionalEncoding(nn.Module):

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 256,
        base_size: Tuple[int, int] = None,
        log_scale: bool = False,
        align_corners: bool = False,
        normalized_interpolation: bool = True,
    ):
        """Basic definition of positional encoding.

        Args:
            input_channels (int): the dimension of input.
            embedding_dim (int, optional): the dimension of embedding.
                Defaults to 256.
            base_size (Tuple[int, int], optional): If it is not None, the
                positional encoding is performed on a small map and
                interpolate to input size. Defaults to None.
        """
        if input_channels != 2:
            warnings.warn('The position cords is not [x,y]. If you are using '
                          'approximated encoding, the other dimension will be '
                          'initialized with zero.')

        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.approximate_mode = base_size is not None
        self.base_size = base_size
        self.log_scale = log_scale
        self.align_corners = align_corners
        self.normalized_interpolation = normalized_interpolation
        if self.base_size is not None:
            self.register_buffer('base_embedding',
                                 self.init_embedding_map(*self.base_size))

    def init_embedding_map(self, w: int, h: int) -> torch.Tensor:
        half_w = w // 2
        half_h = h // 2
        base_position_mat = torch.stack(
            (
                torch.arange(-half_w, half_w + 1).repeat(2 * half_h + 1, 1),
                torch.arange(-half_h, half_h + 1).unsqueeze(1).repeat(
                    1, half_w * 2 + 1),
            ),
            dim=2,
        )  # h,w,2
        if self.input_channels > 2:
            left_embedding_dim = self.input_channels - 2
            base_position_mat = torch.cat(
                [
                    base_position_mat,
                    base_position_mat.new_zeros(h, w, left_embedding_dim),
                ],
                dim=-1,
            )

        return self.preprocess_embedding(base_position_mat[None, ...])

    def pre_compute(self):
        if self.approximate_mode:
            return self.postprocess_embedding(self.base_embedding).permute(
                0, 3, 1, 2)  # 1,C,H,W
        return None

    def forward(self,
                positions: torch.Tensor,
                base_embedding: torch.Tensor = None) -> torch.Tensor:
        """Get the positional embedding for specific positions.

        Args:
            positions (torch.Tensor): [N,H,W,2] or [N,*,input_channels].
                For approximated model, the positions must 2-dim cords in
                shape [N,H,W,2].
        Returns:
            torch.Tensor: [N,*,C]
        """
        if self.approximate_mode:
            # interpolate
            if base_embedding is None:
                base_embedding = self.postprocess_embedding(
                    self.base_embedding).permute(0, 3, 1, 2)  # 1,C,H,W
            if self.normalized_interpolation:
                positions = positions / (0.5 * positions.new_tensor(
                    self.base_size).reshape(1, 1, 1, 2))
            embedding = F.grid_sample(
                base_embedding.expand(
                    positions.size(0),
                    *base_embedding.size()[1:]),
                positions,
                padding_mode='border',
                align_corners=self.align_corners,
            )
            embedding = embedding.permute(0, 2, 3, 1)
        else:
            embedding = self.preprocess_embedding(positions)
            embedding = self.postprocess_embedding(embedding)
        # clamp embedding
        if self.log_scale:
            embedding = torch.log(embedding.clamp(min=1e-6))
        return embedding

    def preprocess_embedding(self, position_mat: torch.Tensor) -> torch.Tensor:
        """Preprocess the input positions such as doing sine, cosine
        transformation.

        Args:
            position_mat (torch.Tensor): [N,*,input_channels]

        Returns:
            torch.Tensor: [N,*,embedding_dim]
        """
        return position_mat

    def postprocess_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Perform learnable operations on the embedding.

        Args:
            embedding (torch.Tensor): [N,*,C]

        Returns:
            torch.Tensor: [N,*,C]
        """
        return embedding


@BVR_POSITIONAL_ENCODING.register_module()
class PositionalEncodingSine(BasePositionalEncoding):

    def __init__(self,
                 input_channels: int,
                 temperature: float = 1000,
                 scale: float = 1.0,
                 normalize=False,
                 **kwargs):
        """Generate sine positional embedding.

        See `BasePositionalEncoding` for details.
        Args:
            input_channels (int): the dimension of input.
            temperature (float, optional): Defaults to 1000.
            scale (float, optional): Defaults to 1.0.
            normalize (bool, optional): Defaults to False.
        """
        self.temperature = temperature
        self.scale = scale
        self.normalize = normalize
        super(PositionalEncodingSine, self).__init__(input_channels, **kwargs)

    def preprocess_embedding(self, position_mat: torch.Tensor) -> torch.Tensor:
        """Preprocess the input positions such as doing sine, cosine
        transformation.

        Args:
            position_mat (torch.Tensor): [N,*,input_channels]

        Returns:
            torch.Tensor: [N,*,embedding_dim]
        """
        if self.normalize:
            bin_scale = position_mat.max() - position_mat.min()
            if self.input_channels > 2:
                cords = position_mat[..., :2] / bin_scale
                position_mat = torch.cat([cords, position_mat[..., 2:]],
                                         dim=-1)
            else:
                position_mat = position_mat / bin_scale

        feat_dim = self.embedding_dim // (2 * self.input_channels)
        feat_range = torch.arange(0, feat_dim, device=position_mat.device)
        dim_mat = torch.pow(1.0 * self.temperature,
                            (4.0 / self.embedding_dim) * feat_range)
        position_mat = self.scale * position_mat.unsqueeze(-1)

        div_mat = torch.div(position_mat,
                            dim_mat)  # [N,*,input_channels,feat_dim]
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        embedding = torch.cat((sin_mat, cos_mat),
                              dim=-1)  # [N,*,input_channels,2*feat_dim]
        embedding = embedding.view(*embedding.size()[:-2], self.embedding_dim)

        return embedding


@BVR_POSITIONAL_ENCODING.register_module()
class PositionalEncodingSineLearned(PositionalEncodingSine):

    def __init__(self,
                 input_channels: int,
                 out_channels: int = None,
                 conv_dim: int = 2,
                 num_layers: int = 1,
                 out_relu: bool = False,
                 norm_cfg: dict = None,
                 **kwargs):
        """Sine postional embedding with learnable transformation.

        Args:
            input_channels (int): the dimension of input.
            with_sigmoid (bool, optional): add sigmoid function on the top of
                output. Defaults to False.
            conv_dim (int, optional): 1 or 2. If it is 2, the input positions
                much be [N,H,W,C]. Defaults to 2.
            norm_cfg (dict, optional): Defaults to None.
        """
        super(PositionalEncodingSineLearned,
              self).__init__(input_channels, **kwargs)
        self.use_conv2d = conv_dim == 2

        if self.use_conv2d:
            conv_cfg = dict(type='Conv2d')
        else:
            conv_cfg = dict(type='Conv1d')
        if out_channels is None:
            out_channels = self.embedding_dim

        layers = []
        for _ in range(num_layers - 1):
            layers.append(
                ConvModule(
                    self.embedding_dim,
                    self.embedding_dim,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ))
        layers.append(
            ConvModule(
                self.embedding_dim,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                act_cfg=None))
        if out_relu:
            layers.append(nn.ReLU())
        self.embedding_transform = nn.Sequential(*layers)

        for m in self.embedding_transform:
            if isinstance(m, ConvModule):
                normal_init(m.conv, std=0.01)

    def postprocess_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Perform learnable operations on the embedding.

        Args:
            embedding (torch.Tensor): [N,*,C]

        Returns:
            torch.Tensor: [N,*,C]
        """
        if self.use_conv2d:
            assert embedding.dim() == 4

            embedding = self.embedding_transform(
                embedding.permute(0, 3, 1,
                                  2).contiguous()).permute(0, 2, 3,
                                                           1)  # [N,C,*]
        else:
            shape = list(embedding.size())
            new_shape = shape[:-1] + [self.out_channels]
            embedding = (
                self.embedding_transform(
                    embedding.reshape(shape[0], -1, shape[-1]).permute(
                        0, 2, 1)).permute(0, 2, 1).reshape(*new_shape))

        return embedding


@BVR_POSITIONAL_ENCODING.register_module()
class PositionalEncodingLearned(BasePositionalEncoding):

    def __init__(self,
                 input_channels: int,
                 with_sigmoid: bool = False,
                 conv_dim: int = 2,
                 norm_cfg: dict = None,
                 **kwargs):
        """Postional embedding with learnable transformation.

        Args:
            input_channels (int): the dimension of input.
            with_sigmoid (bool, optional): add sigmoid function on the top of
                output. Defaults to False.
            conv_dim (int, optional): 1 or 2. If it is 2, the input positions
                much be [N,H,W,C]. Defaults to 2.
            norm_cfg (dict, optional): Defaults to None.
        """
        super(PositionalEncodingLearned,
              self).__init__(input_channels, **kwargs)
        self.use_conv2d = conv_dim == 2
        self.with_sigmoid = with_sigmoid
        if self.use_conv2d:
            conv = nn.Conv2d
            conv_cfg = dict(type='Conv2d')
        else:
            conv = nn.Conv1d
            conv_cfg = dict(type='Conv1d')
        self.embedding_transform = nn.Sequential(
            ConvModule(
                self.input_channels,
                self.embedding_dim,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ),
            conv(self.input_channels, self.embedding_dim, kernel_size=1),
        )
        for m in self.embedding_transform:
            if isinstance(m, ConvModule):
                normal_init(m.conv, std=0.01)
            else:
                normal_init(m, std=0.01)

    def postprocess_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Perform learnable operations on the embedding.

        Args:
            embedding (torch.Tensor): [N,*,C]

        Returns:
            torch.Tensor: [N,*,C]
        """
        if self.use_conv2d:
            assert embedding.dim() == 4

        embedding = self.embedding_transform(
            embedding.transpose(1, -1).contiguous())  # [N,C,*]
        if self.with_sigmoid:
            embedding = embedding.sigmoid()
        return embedding.transpose(1, -1).contiguous()  # [N,*,C]
