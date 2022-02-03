from typing import List

import torch
import torch.nn as nn

from mmdet.core import multi_apply
from ..builder import TRANSFORMER
from .multihead_attention import MultiheadAttention
from .positional_encoding import PositionalEncodingSineLearned


@TRANSFORMER.register_module()
class SimpleBVR_Transformer(nn.Module):
    """Single Layer Tansformer with Self Attention."""

    def __init__(self,
                 position_dim,
                 embedding_dim,
                 num_heads,
                 num_outer_heads=1,
                 outer_agg_type='add',
                 positional_cfg=dict(base_size=[300, 300]),
                 with_relative_positional_encoding=True,
                 with_appearance_relation=True,
                 shared_positional_encoding=True,
                 relative_positional_encoding=None,
                 cat_pos=False):
        super().__init__()
        self.with_relative_positional_encoding = \
            with_relative_positional_encoding

        self.decoder = nn.ModuleList()
        if self.with_relative_positional_encoding:
            self.relative_positional_encoding = nn.ModuleList()
            if relative_positional_encoding is not None:
                shared_positional_encoding = True
        for _ in range(num_outer_heads):
            self.decoder.append(
                MultiheadAttention(
                    embedding_dim,
                    num_heads,
                    dropout=0,
                    app_relation=with_appearance_relation))
            if self.with_relative_positional_encoding and (
                    not shared_positional_encoding):
                self.relative_positional_encoding.append(
                    PositionalEncodingSineLearned(
                        position_dim,
                        out_channels=num_heads,
                        conv_dim=2,
                        embedding_dim=embedding_dim,
                        **positional_cfg,
                    ))
        if (self.with_relative_positional_encoding
                and shared_positional_encoding):
            if relative_positional_encoding is not None:
                self.relative_positional_encoding = \
                    relative_positional_encoding
            else:
                self.relative_positional_encoding = \
                    PositionalEncodingSineLearned(
                        position_dim,
                        out_channels=num_heads,
                        conv_dim=2,
                        embedding_dim=embedding_dim,
                        **positional_cfg)
        self.cat_pos = cat_pos
        if self.cat_pos:
            self.embed_trans = nn.Sequential(
                nn.Linear(embedding_dim + num_outer_heads * position_dim,
                          embedding_dim), nn.ReLU())
            self.pos_trans = nn.Linear(position_dim, position_dim, bias=False)
            nn.init.normal_(self.embed_trans[0].weight, 0.01)

        self.num_outer_heads = num_outer_heads
        self.outer_agg_type = outer_agg_type
        self.out_act = nn.ReLU()

    def forward(
        self,
        query_features: List[torch.Tensor],
        query_positions: List[torch.Tensor],
        key_features: List[List[torch.Tensor]],
        key_positions: List[List[torch.Tensor]],
        scale_terms: List[int] = None,
    ) -> List[torch.Tensor]:
        """Perform SelfAttention on features.

        Args:
            query_features (List[torch.Tensor]):
                each tensor has shape [N,H,W,C]
            query_positions (List[torch.Tensor]):
                each tensor has shape [N,H,W,2]
            key_features (List[List[torch.Tensor]]):
                each tensor has shape [N,K,C]
            key_positions (List[List[torch.Tensor]]):
                each tensor has shape [N,K,2]
            scale_terms (List[int]): scale factor for positions.
        Returns:
            List[torch.Tensor]: [description]
        """
        if scale_terms is None:
            scale_terms = [1.0] * len(query_features)
        elif isinstance(scale_terms, float):
            scale_terms = [scale_terms] * len(query_features)
        else:
            assert isinstance(scale_terms, list) and (len(scale_terms)
                                                      == len(query_features))
        # For each level, each kind of keypoints,
        # we only compute embedding basis(400x400) once.
        if self.with_relative_positional_encoding:
            # precompute positional embedding if approximated
            if isinstance(self.relative_positional_encoding, nn.ModuleList):
                base_embedding = []
                for m in self.relative_positional_encoding:
                    base_embedding.append(m.pre_compute())
            else:
                base_embedding = [
                    self.relative_positional_encoding.pre_compute()
                ] * self.num_outer_heads
        else:
            base_embedding = None

        return multi_apply(
            self.forward_single,
            query_features,
            query_positions,
            key_features,
            key_positions,
            scale_terms,
            base_embedding=base_embedding,
        )

    def forward_single(
        self,
        query_feature,
        query_position,
        key_features,
        key_positions,
        scale_terms,
        base_embedding,
    ):
        input_size = list(query_feature.size())

        N = query_feature.size(0)
        C = query_feature.size(-1)
        query_feature = query_feature.reshape(N, -1, C)  # N,HW,C
        query_position = (query_position.reshape(N, -1, 2) / scale_terms
                          )  # scale the position

        query_new_feature = []
        query_new_pos = []
        # each group represents a kind of keypoint
        for group in range(self.num_outer_heads):
            key_feature = key_features[group].reshape(N, -1, C)  # N,K,C
            key_position = key_positions[group].reshape(
                N, -1, 2) / scale_terms  # N,K,2
            rel_pos = key_position[:, None, ...] - query_position[..., None, :]
            if self.with_relative_positional_encoding:
                embedding_layer = (
                    self.relative_positional_encoding[group] if isinstance(
                        self.relative_positional_encoding, nn.ModuleList) else
                    self.relative_positional_encoding)
                relative_positional_encoding = embedding_layer(
                    rel_pos,
                    base_embedding[group],
                )  # N,HW,K,C
                relative_positional_encoding = \
                    relative_positional_encoding.permute(0, 3, 1, 2)
                # N,C,HW,K
            else:
                relative_positional_encoding = None

            refined_feat, attn_weight = self.decoder[group](
                query_feature.permute(1, 0, 2),  # Len,Batch,Channel
                key_feature.permute(1, 0, 2),  # Len,Batch,Channel
                key_feature.permute(1, 0, 2),  # Len,Batch,Channel
                need_weights=True,
                relative_atten_weights=relative_positional_encoding,
            )
            if self.cat_pos:
                new_rel_pos = (attn_weight[..., None] *
                               rel_pos.detach()).sum(dim=-2)
                query_new_pos.append(self.pos_trans(new_rel_pos))
            query_new_feature.append(refined_feat)

        if self.outer_agg_type == 'add':
            query_feature = query_feature + torch.stack(query_new_feature).sum(
                dim=0).permute(1, 0, 2)
        else:
            raise NotImplementedError()
        if self.out_act:
            query_feature = self.out_act(query_feature)
        if self.cat_pos:
            query_feature = self.embed_trans(
                torch.cat([query_feature] + query_new_pos, dim=-1))
        query_feature = query_feature.reshape(*input_size)
        return query_feature, None
