from typing import Dict, List, Tuple, Union

import torch
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.utils import BRPool, TLPool
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class KeypointHead(AnchorFreeHead):
    """Predict keypoints of object.

    Args:
        num_classes (int): category numbers of objects in dataset.
        in_channels (int): Dimension of input features.
        shared_stacked_convs (int): Number of shared conv layers for all
            keypoint heads.
        logits_convs (int): Number of conv layers for each logits.
        head_types (List[str], optional): Number of head. Each head aims to
            predict different type of keypoints. Defaults to
            ["top_left_corner", "bottom_right_corner", "center"].
        corner_pooling (bool): Whether to use corner pooling for corner
            keypoint prediction. Defaults to False.
        loss_offset (dict, optional): Loss configuration for keypoint
            offset prediction. Defaults to dict(type='SmoothL1Loss',
            loss_weight=1.0/9.0).
        **kwargs:
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 shared_stacked_convs: int = 0,
                 logits_convs: int = 0,
                 head_types=None,
                 corner_pooling: bool = False,
                 loss_offset=None,
                 **kwargs) -> None:
        if loss_offset is None:
            loss_offset = dict(type='SmoothL1Loss', loss_weight=1.0 / 9.0)
        if head_types is None:
            head_types = ['top_left_corner', 'bottom_right_corner', 'center']
        self.corner_pooling = corner_pooling
        self.shared_stacked_convs = shared_stacked_convs
        self.logits_convs = logits_convs
        self.head_types = head_types
        super(KeypointHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_offset = build_loss(loss_offset)
        if self.train_cfg is not None:
            self.point_assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self) -> None:
        """Construct the model."""
        # construct shared layers
        self.shared_layers = self._init_layer_list(self.in_channels,
                                                   self.shared_stacked_convs)
        # construct separated heads
        self.keypoint_layers = nn.ModuleDict()
        self.keypoint_cls_heads = nn.ModuleDict()
        self.keypoint_offset_heads = nn.ModuleDict()

        in_channels = (
            self.in_channels
            if self.shared_stacked_convs == 0 else self.feat_channels)

        for head_type in self.head_types:
            keypoint_layer = self._init_layer_list(in_channels,
                                                   self.stacked_convs)
            if 'corner' in head_type and self.corner_pooling:
                if 'top_left' in head_type:
                    keypoint_layer.append(
                        TLPool(
                            self.feat_channels,
                            self.conv_cfg,
                            self.norm_cfg,
                            3,
                            1,
                            corner_dim=64))
                else:
                    keypoint_layer.append(
                        BRPool(
                            self.feat_channels,
                            self.conv_cfg,
                            self.norm_cfg,
                            3,
                            1,
                            corner_dim=64))
            self.keypoint_layers.update({head_type: keypoint_layer})

            # head
            keypoint_cls_head = self._init_layer_list(self.feat_channels,
                                                      self.logits_convs)
            keypoint_cls_head.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes,
                    3,
                    stride=1,
                    padding=1))
            self.keypoint_cls_heads.update({head_type: keypoint_cls_head})

            keypoint_offset_head = self._init_layer_list(
                self.feat_channels, self.logits_convs)
            keypoint_offset_head.append(
                nn.Conv2d(self.feat_channels, 2, 3, stride=1, padding=1))
            self.keypoint_offset_heads.update(
                {head_type: keypoint_offset_head})

    def _init_layer_list(self, in_channels: int,
                         num_convs: int) -> nn.ModuleList:
        """
        Args:
            in_channels (int):
            num_convs (int):
        """
        layers = nn.ModuleList()
        for i in range(num_convs):
            chn = in_channels if i == 0 else self.feat_channels
            layers.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return layers

    def init_weights(self):

        def init_to_apply(m):
            if isinstance(m, ConvModule):
                normal_init(m.conv, std=0.01)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                normal_init(m, std=0.01)

        for layer in self.shared_layers:
            normal_init(layer.conv, std=0.01)
        for _, layer in self.keypoint_layers.items():
            for m in layer:
                if isinstance(m, ConvModule):
                    normal_init(m.conv, std=0.01)
                else:
                    m.apply(init_to_apply)
        bias_cls = bias_init_with_prob(0.01)
        for _, head in self.keypoint_cls_heads.items():
            for i, m in enumerate(head):
                if i != len(head) - 1:
                    normal_init(m.conv, std=0.01)
                else:
                    normal_init(m, std=0.01, bias=bias_cls)
        for _, head in self.keypoint_offset_heads.items():
            for i, m in enumerate(head):
                if i != len(head) - 1:
                    normal_init(m.conv, std=0.01)
                else:
                    normal_init(m, std=0.01)

    def forward(
        self,
        feats: List[torch.Tensor],
        choices: Union[str, List[str]] = None
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """Predict the keypoint and return category and offset.

        Args:
            feats (List[torch.Tensor]): feature map lists. Each is [N,C,Hi,Wi].
            choices (Union[str,List[str]], optional): Select which head to use.

        Returns:
            Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]]: [description]
        """
        if choices is None:
            choices = self.head_types
        elif isinstance(choices, str):
            choices = [choices]
        keypoint_pred = multi_apply(
            self.forward_single, feats, choices=choices)

        kp_scores = keypoint_pred[:len(choices)]
        kp_offsets = keypoint_pred[len(choices):]
        ch2scores = {ch: scores for ch, scores in zip(choices, kp_scores)}
        ch2offsets = {ch: offsets for ch, offsets in zip(choices, kp_offsets)}
        return ch2scores, ch2offsets

    def forward_single(self, x: torch.Tensor,
                       choices: List[str]) -> Tuple[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [N,C,H,W]. Input Features.
            choices (List[str]): names of head to use.
        Returns:
            Tuple[torch.Tensor]: head_0_score,...,head_`len(choice)`_score,
                head_0_offset,...head_`len(choice)`_offset
        """
        feat = x
        for layer in self.shared_layers:
            feat = layer(feat)

        keypoint_offsets = []
        keypoint_clses = []
        for head_type in choices:
            keypoint_feat = feat
            for layer in self.keypoint_layers[head_type]:
                keypoint_feat = layer(keypoint_feat)
            offset_feat = cls_feat = keypoint_feat
            for layer in self.keypoint_cls_heads[head_type]:
                cls_feat = layer(cls_feat)
            for layer in self.keypoint_offset_heads[head_type]:
                offset_feat = layer(offset_feat)
            keypoint_clses.append(cls_feat)
            keypoint_offsets.append(offset_feat)

        return tuple(keypoint_clses) + tuple(keypoint_offsets)

    def _get_targets_single(
            self, gt_points: torch.Tensor, gt_bboxes: torch.Tensor,
            gt_labels: torch.Tensor, points: torch.Tensor,
            num_points: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute targets for single image.

        Args:
            gt_points (torch.Tensor): Ground truth points for single image with
                shape (num_gts, 2) in [x, y] format.
            gt_bboxes (torch.Tensor): Ground truth bboxes of single image, each
                has shape (num_gt, 4).
            gt_labels (torch.Tensor): Ground truth labels of single image, each
                has shape (num_gt,).
            points (torch.Tensor): Points for all level with shape (num_points,
                3) in [x,y,stride] format.
            num_points (List[int]): Points num for each level.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        assigner = self.point_assigner
        offset_target, score_target, pos_mask = assigner.assign(
            points, num_points, gt_points, gt_bboxes, gt_labels,
            self.num_classes)
        return score_target, offset_target, pos_mask[:, None]

    def get_targets(
        self, points: List[torch.Tensor], gt_points_list: List[torch.Tensor],
        gt_bboxes_list: List[torch.Tensor], gt_labels_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (List[torch.Tensor]): Points for each level with shape
                (num_points, 3) in [x,y,stride] format.
            gt_points_list (List[torch.Tensor]): Ground truth points for each
                image with shape (num_gts, 2) in [x, y] format.
            gt_bboxes_list (List[torch.Tensor]): Ground truth bboxes of each
                image, each has shape (num_gt, 4).
            gt_labels_list (List[torch.Tensor]): Ground truth labels of each
                box, each has shape (num_gt,).

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: score targets and offset targets
            and positive_mask for all images, each has shape [batch,
            num_points, channel].
        """
        num_points = [point.size()[0] for point in points]
        points = torch.cat(points, dim=0)
        score_target_list, offset_target_list, pos_mask_list = multi_apply(
            self._get_targets_single,
            gt_points_list,
            gt_bboxes_list,
            gt_labels_list,
            points=points,
            num_points=num_points)
        return (torch.stack(score_target_list),
                torch.stack(offset_target_list), torch.stack(pos_mask_list))

    @force_fp32(apply_to=('keypoint_scores', 'keypoint_offsets'))
    def loss(self, keypoint_scores: List[torch.Tensor],
             keypoint_offsets: List[torch.Tensor], keypoint_types: List[str],
             gt_points: List[torch.Tensor], gt_bboxes: List[torch.Tensor],
             gt_labels: List[torch.Tensor],
             img_metas: List[dict]) -> Dict[str, torch.Tensor]:
        """Compute loss of single head. Note: For multiple head, we propose to
        concatenate the tensor along batch dimension to speed up this process.

        Args:
            keypoint_scores (List[torch.Tensor]): keypoint scores for each
                level for each head.
            keypoint_offsets (List[torch.Tensor]): keypoint offsets for each
                level for each head.
            keypoint_types: List[str]: The types of keypoint heads.
            gt_points (List[torch.Tensor]): Ground truth points for each image
                with shape (num_gts, 2) in [x, y] format.
            gt_bboxes (List[torch.Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (List[torch.Tensor]): class indices corresponding to each
                box.
            img_metas (List[dict]): Meta information of each image, e.g., image
                size, scaling factor, etc.

        Returns:
            Dict[str,torch.Tensor]: Loss for head
        """
        featmap_sizes = [score.size()[-2:] for score in keypoint_scores]
        points = self.get_points(featmap_sizes, gt_points[0].dtype,
                                 gt_points[0].device)
        keypoint_scores = _flatten_concat(keypoint_scores).permute(
            0, 2, 1)  # [batch,num_points,num_classes]
        keypoint_offsets = _flatten_concat(keypoint_offsets).permute(
            0, 2, 1)  # [batch,num_points,2]
        score_targets, offset_targets, pos_masks = self.get_targets(
            points, gt_points, gt_bboxes, gt_labels)

        avg_factor = reduce_mean(torch.sum(pos_masks))
        # TODO: Maybe positive samples and negative samples should have
        # different avg factors.
        loss_cls = self.loss_cls(
            keypoint_scores.sigmoid(), score_targets, avg_factor=avg_factor)
        loss_offset = self.loss_offset(
            keypoint_offsets,
            offset_targets,
            weight=pos_masks.expand_as(keypoint_offsets),
            avg_factor=avg_factor)
        return {'loss_point_cls': loss_cls, 'loss_point_offset': loss_offset}

    def loss_multihead(self, keypoint_scores: Dict[str, List[torch.Tensor]],
                       keypoint_offsets: Dict[str, List[torch.Tensor]],
                       gt_bboxes: List[torch.Tensor],
                       gt_labels: List[torch.Tensor],
                       img_metas: List[dict]) -> Dict[str, torch.Tensor]:
        """Compute loss of multiple heads. :param keypoint_scores: keypoint
        scores for each level for each head. :type keypoint_scores: Dict[str,
        List[torch.Tensor]] :param keypoint_offsets: keypoint offsets for each
        level for each head. :type keypoint_offsets: Dict[str,
        List[torch.Tensor]] :param gt_bboxes: Ground truth bboxes for each
        image with.

            shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

        Args:
            keypoint_scores:
            keypoint_offsets:
            gt_bboxes:
            gt_labels (List[torch.Tensor]): class indices corresponding to each
                box.
            img_metas (List[dict]): Meta information of each image, e.g., image
                size, scaling factor, etc.

        Returns:
            Dict[str,torch.Tensor]: Loss for head
        """
        # TODO: check the order of concated tensor
        names, keypoint_scores = _concat(keypoint_scores)
        _, keypoint_offsets = _concat(keypoint_offsets)
        gt_points = self._box2point(
            names, gt_bboxes)  # keypoint_type*batch*[num_gt,2]
        return self.loss(keypoint_scores, keypoint_offsets, names, gt_points,
                         gt_bboxes * len(names), gt_labels * len(names),
                         img_metas * len(names))

    def get_keypoints_single(self,
                             keypoint_logits: torch.Tensor,
                             keypoint_offsets: torch.Tensor,
                             locations: torch.Tensor,
                             stride: int,
                             max_keypoint_num: int = 20,
                             keypoint_score_thr: float = 0.1,
                             block_grad: bool = False):
        """Extract keypoints from a sinle heat map.

        Args:
            keypoint_logits (torch.Tensor): [N,C,H,W]
            keypoint_offsets (torch.Tensor): [N,2,H,W]
            locations (torch.Tensor): [H*W,3]
            stride (int): Resolution of current feature map.
            max_keypoint_num (int, optional): Maximum keypoints to extract.
                Defaults to 20.
            keypoint_score_thr (float, optional): Keypoints which are below
                this threshold are ignored. Not used. Defaults to 0.1.
            block_grad (bool, optional): Whether to block the gradient of the
                extraction process. Defaults to False.
        """

        def _local_nms(heatmap: torch.Tensor,
                       kernel_size: int = 3) -> torch.Tensor:
            """Find the local maximum points of a heatmap.

            Args:
                heatmap (torch.Tensor): Shape is [N,C,H,W].
                kernel_size (int): the size of kernel used i nms.
                    Defaults to 3.

            Returns:
                torch.Tensor: heatmap with score only on local maximum points.
            """
            pad = (kernel_size - 1) // 2
            hmax = F.max_pool2d(
                heatmap, (kernel_size, kernel_size), stride=1, padding=pad)
            keep = (hmax == heatmap).float()
            return heatmap * keep

        keypoint_scores = _local_nms(keypoint_logits.sigmoid())
        topk_score, topk_inds, _, topk_ys, topk_xs = _topk(
            keypoint_scores, locations, max_keypoint_num)
        topk_offsets = _gather_feat(
            keypoint_offsets.reshape(keypoint_offsets.size(0), 2,
                                     -1).permute(0, 2, 1), topk_inds)
        if block_grad:
            topk_offsets = topk_offsets.detach()
            topk_score = topk_score.detach()
        keypoint_pos_round = torch.stack([topk_xs, topk_ys], dim=-1)
        keypoint_pos = keypoint_pos_round + topk_offsets * stride
        return topk_score, keypoint_pos, topk_inds

    def get_keypoints(
        self,
        keypoint_logits: List[torch.Tensor],
        keypoint_offsets: List[torch.Tensor],
        max_keypoint_num: int = 20,
        keypoint_score_thr: float = 0.1,
        block_grad: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
               List[torch.Tensor]]:
        """Extract keypoints for single head.

        Note: For multiple head, we
        propose to concatenate the tensor along batch dimension to speed up
        this process. We do not implement this function for multiple heads as
        little operation is needed for that purpose.
        Args:
            keypoint_scores (List[torch.Tensor]): keypoint scores for each
                level.
            keypoint_offsets (List[torch.Tensor]): keypoint offsets for each
                level.
            keypoint_features (List[torch.Tensor]): featuremap to select
                features for each level.
            max_keypoint_num (int): maximum number of selected keypoints.
                Defaults to 20.
            keypoint_score_thr (float): keypoints with score below this terms
                are discarded.
        Returns:
            Tuple[List[torch.Tensor],List[torch.Tensor]]: Keypoint scores and
                positions for each level.
                Each score tensor has shape [batch,max_keypoint_num]. Each
                position tensor has shape [batch,max_keypoint_num,3] in which
                the last dimension indicates [x,y,category].
        """
        featmap_sizes = [hm.size()[-2:] for hm in keypoint_logits]
        points = self.get_points(featmap_sizes, keypoint_logits[0].dtype,
                                 keypoint_logits[0].device)
        keypoint_scores, keypoint_pos, keypoint_inds = multi_apply(
            self.get_keypoints_single,
            keypoint_logits,
            keypoint_offsets,
            points,
            self.strides,
            max_keypoint_num=max_keypoint_num,
            keypoint_score_thr=keypoint_score_thr,
            block_grad=block_grad)
        return keypoint_scores, keypoint_pos, keypoint_inds, points

    def get_keypoints_multihead(
        self,
        keypoint_logits: Dict[str, List[torch.Tensor]],
        keypoint_offsets: Dict[str, List[torch.Tensor]],
        keypoint_choices: List[str],
        map_back: bool = True,
        **kwargs
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]],
               List[List[torch.Tensor]], List[List[torch.Tensor]], ]:
        """Extract Keypoints and Return Absolute Position of all keypoints.
        See `get_keypoints`
        Args:
            keypoint_logits (Dict[str, List[torch.Tensor]]): [description]
            keypoint_offsets (Dict[str, List[torch.Tensor]]): [description]
            keypoint_choices (List[str]): [description]
            map_back (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        names, keypoint_logits = _concat(
            {ch: keypoint_logits[ch]
             for ch in keypoint_choices},
            index=keypoint_choices)
        _, keypoint_offsets = _concat(
            {ch: keypoint_offsets[ch]
             for ch in keypoint_choices},
            index=keypoint_choices)

        keypoint_scores, keypoint_pos, keypoint_inds, locations = \
            self.get_keypoints(keypoint_logits, keypoint_offsets, **kwargs)
        if map_back:
            keypoint_scores = _split(keypoint_scores, names)
            keypoint_pos = _split(keypoint_pos, names)
            keypoint_inds = _split(keypoint_inds, names)

        return keypoint_scores, keypoint_pos, keypoint_inds, locations

    def get_keypoint_features(self,
                              feature_sets,
                              keypoint_scores,
                              keypoint_positions,
                              keypoint_inds,
                              num_keypoint_head=1,
                              selection_method='index',
                              cross_level_topk=-1,
                              cross_level_selection=False):
        # h,w -> w,h
        image_size = list(feature_sets[0].size())[-2:][::-1]
        image_size = [imsize * self.strides[0] for imsize in image_size]

        def _feature_selection(featuremaps: torch.Tensor,
                               sample_positions: torch.Tensor,
                               sample_inds: torch.Tensor = None):
            """

            Args:
                featuremaps (torch.Tensor): [N,C,H,W]
                sample_positions (torch.Tensor): [N,K,2]
                sample_inds (torch.Tensor): [N,K]
            """
            if selection_method == 'index':
                if sample_inds is None:
                    H, W = featuremaps.size()[-2:]
                    downsample_scale = torch.sqrt(
                        (image_size[0] * image_size[1]) / (H * W))
                    sample_inds = (sample_positions[:, 1] * W +
                                   sample_positions[:, 0]) / downsample_scale
                    sample_inds = torch.floor(sample_inds).long()
                featuremaps = featuremaps.reshape(*featuremaps.size()[:2],
                                                  -1).permute(0, 2, 1)
                if featuremaps.size(0) != sample_inds.size(0):
                    assert sample_inds.size(0) % featuremaps.size(0) == 0
                    featuremaps = (
                        featuremaps[None, ...].expand(
                            sample_inds.size(0) // featuremaps.size(0), -1, -1,
                            -1).reshape(-1,
                                        *featuremaps.size()[-2:]))
                return _gather_feat(featuremaps, sample_inds)
            elif selection_method == 'interpolation':
                grid = (
                    sample_positions * 2.0 /
                    sample_positions.new_tensor(image_size).reshape(1, 1, 2) -
                    1.0)
                # assert grid.max()<=1
                if featuremaps.size(0) != grid.size(0):
                    assert grid.size(0) % featuremaps.size(0) == 0
                    featuremaps = (
                        featuremaps[None, ...].expand(
                            grid.size(0) // featuremaps.size(0), -1, -1, -1,
                            -1).reshape(-1,
                                        *featuremaps.size()[1:]))
                return (F.grid_sample(
                    featuremaps,
                    grid.unsqueeze(1),
                    align_corners=False,
                    padding_mode='border').squeeze(2).permute(0, 2, 1))
            else:
                raise NotImplementedError()

        if cross_level_topk > 0:
            # rerank across all level
            all_level_scores: torch.Tensor = torch.cat(keypoint_scores, dim=-1)
            # rank
            _, topk_inds = torch.topk(
                all_level_scores, k=cross_level_topk, dim=-1)
            all_level_positions: torch.Tensor = torch.cat(
                keypoint_positions, dim=1)

            topk_positions = _gather_feat(all_level_positions, topk_inds)
            if cross_level_selection:
                # select on each level
                keypoint_features: List[torch.Tensor] = [
                    _feature_selection(feature_sets[i], keypoint_positions[i],
                                       keypoint_inds[i])
                    for i in range(len(feature_sets))
                ]
                topk_features = _gather_feat(
                    torch.cat(keypoint_features, dim=1), topk_inds)
                keypoint_features = [topk_features] * len(keypoint_scores)
            else:
                keypoint_features: List[torch.Tensor] = [
                    _feature_selection(feature_sets[i], topk_positions)
                    for i in range(len(feature_sets))
                ]
            keypoint_positions = [topk_positions] * len(keypoint_scores)
        else:
            # select on each level
            keypoint_features: List[torch.Tensor] = [
                _feature_selection(feature_sets[i], keypoint_positions[i],
                                   keypoint_inds[i])
                for i in range(len(feature_sets))
            ]

        return [
            keypoint_feature.chunk(num_keypoint_head, dim=0)
            for keypoint_feature in keypoint_features
        ], [
            keypoint_position.chunk(num_keypoint_head, dim=0)
            for keypoint_position in keypoint_positions
        ]

    @staticmethod
    def _box2point(point_types: List[str],
                   boxes: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extract keypoints from bboxes.

        Args:
            point_types (List[str]): types for keypoint to extract.
            boxes (List[torch.Tensor]): bboxes for each image with shape
                (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

        Returns:
            points (List[torch.Tensor]): points for each type and each image
            with shape (num_gts,2) in [x,y] format.
        """
        points = []
        for point_type in point_types:
            if point_type == 'top_left_corner':
                points.extend(
                    [boxes[img_i][:, :2] for img_i in range(len(boxes))])
            elif point_type == 'bottom_right_corner':
                points.extend(
                    [boxes[img_i][:, 2:] for img_i in range(len(boxes))])
            elif point_type == 'center':
                points.extend([
                    boxes[img_i][:, :2] * 0.5 + boxes[img_i][:, 2:] * 0.5
                    for img_i in range(len(boxes))
                ])
        return points

    def get_points(
        self,
        featmap_sizes: List[Tuple[int, int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (List[Tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            List[torch.Tensor]: points for all levels in each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            y, x = self._get_points_single(featmap_sizes[i], self.strides[i],
                                           dtype, device, True)
            y = y * self.strides[i] + self.strides[i] // 2
            x = x * self.strides[i] + self.strides[i] // 2
            mlvl_points.append(
                torch.stack([x, y, x.new_full(x.size(), self.strides[i])],
                            dim=1))
        return mlvl_points

    def get_bboxes(self, keypoint_scores: Dict[str, List[torch.Tensor]],
                   keypoint_offsets: Dict[str, List[torch.Tensor]]):
        """Get boxes. We will not use this function in our project.

        Args:
            keypoint_scores (Dict[str, List[torch.Tensor]]): keypoint scores
                for each level for each head.
            keypoint_offsets (Dict[str, List[torch.Tensor]]): keypoint offsets
                for each level for each head.
        """
        raise NotImplementedError()


def _concat(tensors: Dict[str, List[torch.Tensor]],
            index: List[str] = None) -> Tuple[List[str], List[torch.Tensor]]:
    """Concat tensor dict and return their keys, concatenated values.

    Args:
        tensors (Dict[str, List[torch.Tensor]]):
        index (List[str]): Optional.
    """
    if index:
        names = index
    else:
        names = list(tensors.keys())
    return names, [
        torch.cat(values, dim=0)
        for values in zip(*[tensors[name] for name in names])
    ]


def _split(tensors: List[torch.Tensor],
           keys: List[str]) -> Dict[str, List[torch.Tensor]]:
    """Rearange tensor list to tensor dict.

    Args:
        tensors (List[torch.Tensor]): [description]
        keys (List[str]): [description]

    Returns:
        Dict[str, List[torch.Tensor]]: [description]
    """
    num_rep = len(keys)
    num_batch = tensors[0].size(0) // num_rep
    return {
        keys[i]:
        [tensor[num_batch * i:num_batch * (i + 1)] for tensor in tensors]
        for i in range(len(keys))
    }


def _flatten_concat(tensor_list: List[torch.Tensor]):
    """Flatten tensors and concatenate them together.

    Args:
        tensor_list (List[torch.Tensor]):  List[[N,C,H,W]]
    """
    return torch.cat(
        [
            tensor.reshape(tensor.size(0), tensor.size(1), -1)
            for tensor in tensor_list
        ],
        dim=-1,
    )


def _gather_feat(feat: torch.Tensor,
                 ind: torch.Tensor,
                 mask: torch.Tensor = None):
    """Select features with spatial inds.

    Args:
        feat (torch.Tensor): [N,K,C]
        ind (torch.Tensor): [N,M]
        mask (torch.Tensor): [N,M]. Defaults to None.

    Returns:
        feat (torch.Tensor): [N,M,C]
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores: torch.Tensor, locations: torch.Tensor, k: int = 40):
    """Find the topk points in a map.

    Args:
        scores (torch.Tensor): Shape is [N,C,H,W].
        locations (torch.Tensor): Shape is [H*W,3].
        k (int): [description]. Defaults to 40.

    Returns:
        [type]: [description]
    """
    batch, cat, height, width = scores.size()
    pnum = height * width

    topk_scores, topk_inds = torch.topk(
        scores.view(batch, cat, -1), min(pnum, k))
    topk_inds = topk_inds % (height * width)
    topk_score, topk_ind = torch.topk(
        topk_scores.view(batch, -1), min(pnum, k))
    topk_clses = topk_ind // topk_scores.size()[-1]

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),
                             topk_ind).view(batch,
                                            topk_score.size()[-1])
    topk_locations = _gather_feat(
        locations[None, ...].expand(batch, height * width, 3), topk_inds)
    topk_ys = topk_locations[:, :, 1]
    topk_xs = topk_locations[:, :, 0]

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
