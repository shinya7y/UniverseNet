from mmdet.models.utils.bvr_transformer import SimpleBVR_Transformer
from mmdet.models.utils.bvr_utils import assign_methods_for_bvr
from ..builder import HEADS, build_head
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class BVRHead(BaseDenseHead, BBoxTestMixin):
    """BVRHead for single stage detector.

    `RelationNet++: Bridging Visual Representations for Object Detection via
    Transformer Decoder <https://arxiv.org/abs/2010.15831>`_.

    Args:
        bbox_head_cfg (dict): [description]
        keypoint_head_cfg (dict): [description]
        cls_keypoint_cfg (dict, optional): [description]. Defaults to None.
        reg_keypoint_cfg (dict, optional): [description]. Defaults to None.
        keypoint_pos (str, optional): [description]. Defaults to "input".
        keypoint_cfg (dict, optional): [description]. Defaults to None.
        feature_selection_cfg (dict, optional): [description].
            Defaults to None.
        num_attn_heads (int, optional): [description]. Defaults to 8.
        scale_position (bool, optional): [description]. Defaults to True.
        scale_factor (float, optional): [description]. Defaults to 1.0.
        with_relative_positional_encoding (bool, optional): [description].
            Defaults to True.
        with_appearance_relation (bool, optional): [description].
            Defaults to True.
        shared_positional_encoding_inter (bool, optional): [description].
            Defaults to True.
        shared_positional_encoding_outer (bool, optional): [description].
            Defaults to False.
        reg_cat_pos (bool, optional): [description]. Defaults to False.
        pos_cfg (dict, optional): [description].
            Defaults to dict(base_size=[400, 400]).
        train_cfg (dict, optional): [description]. Defaults to None.
        test_cfg (dict, optional): [description]. Defaults to None.

    Raises:
        RuntimeError: [description]
    """

    def __init__(self,
                 bbox_head_cfg: dict,
                 keypoint_head_cfg: dict,
                 cls_keypoint_cfg: dict = None,
                 reg_keypoint_cfg: dict = None,
                 keypoint_pos: str = 'input',
                 keypoint_cfg: dict = None,
                 feature_selection_cfg: dict = None,
                 num_attn_heads: int = 8,
                 scale_position: bool = True,
                 scale_factor: float = 1.0,
                 with_relative_positional_encoding: bool = True,
                 with_appearance_relation: bool = True,
                 shared_positional_encoding_inter: bool = True,
                 shared_positional_encoding_outer: bool = False,
                 reg_cat_pos: bool = False,
                 pos_cfg: dict = dict(base_size=[400, 400]),
                 train_cfg: dict = None,
                 test_cfg: dict = None):
        super().__init__()
        if keypoint_cfg is None:
            keypoint_cfg = dict(
                max_keypoint_num=20,
                keypoint_score_thr=0.0,
                fuse_multi_level=False,
                max_fuse_keypoint_num=50)
        if feature_selection_cfg is None:
            feature_selection_cfg = dict()
        if reg_keypoint_cfg is None:
            reg_keypoint_cfg = dict(
                keypoint_types=['top_left_corner', 'bottom_right_corner'],
                with_key_score=False,
                with_relation=False)
        if cls_keypoint_cfg is None:
            cls_keypoint_cfg = dict(
                keypoint_types=['center'],
                with_key_score=False,
                with_relation=False)
        if train_cfg is not None:
            bbox_head_cfg.update(train_cfg=train_cfg.bbox)
        if test_cfg is not None:
            bbox_head_cfg.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head_cfg)
        # modify bbox_head by assigning methods
        assign_methods_for_bvr(self.bbox_head)
        if train_cfg is not None:
            keypoint_head_cfg.update(train_cfg=train_cfg.keypoint)
        self.keypoint_head = build_head(keypoint_head_cfg)
        self.keypoint_cfg = keypoint_cfg
        self.feature_selection_cfg = feature_selection_cfg
        self.cls_keypoint_cfg = cls_keypoint_cfg
        self.reg_keypoint_cfg = reg_keypoint_cfg
        self.keypoint_pos = ['input', 'cls', 'reg'].index(keypoint_pos)
        self.scale_position = scale_position
        self.scale_factor = scale_factor
        self.num_heads = num_attn_heads
        self.pos_cfg = pos_cfg
        self.with_relative_positional_encoding = \
            with_relative_positional_encoding
        self.with_appearance_relation = with_appearance_relation
        self.shared_positional_encoding_inter = \
            shared_positional_encoding_inter
        self.shared_positional_encoding_outer = \
            shared_positional_encoding_outer
        self.cat_pos = reg_cat_pos
        self.build_transformer()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # print(self)

    def build_transformer(self):
        if self.cls_keypoint_cfg['with_relation']:
            self.cls_transformer = SimpleBVR_Transformer(
                2,
                self.bbox_head.feat_channels,
                self.num_heads,
                len(self.cls_keypoint_cfg['keypoint_types']),
                with_relative_positional_encoding=self.
                with_relative_positional_encoding,
                with_appearance_relation=self.with_appearance_relation,
                positional_cfg=self.pos_cfg,
                shared_positional_encoding=self.
                shared_positional_encoding_inter)
        if self.reg_keypoint_cfg['with_relation']:
            self.reg_transformer = SimpleBVR_Transformer(
                2,
                self.bbox_head.feat_channels,
                self.num_heads,
                len(self.reg_keypoint_cfg['keypoint_types']),
                positional_cfg=self.pos_cfg,
                with_relative_positional_encoding=self.
                with_relative_positional_encoding,
                with_appearance_relation=self.with_appearance_relation,
                shared_positional_encoding=self.
                shared_positional_encoding_inter,
                relative_positional_encoding=self.cls_transformer.
                relative_positional_encoding
                if self.shared_positional_encoding_outer else None,
                cat_pos=self.cat_pos)

    def init_weights(self):
        self.bbox_head.init_weights()
        self.keypoint_head.init_weights()
        # init transformer

    def forward(self, feats):
        cls_feats, reg_feats = self.bbox_head.forward_features(feats)
        keypoint_scores, keypoint_offsets = self.keypoint_head(
            [feats, cls_feats, reg_feats][self.keypoint_pos])
        if self.cls_keypoint_cfg['with_relation']:
            cls_feats = self._apply_relation(
                cls_feats, cls_feats, keypoint_scores, keypoint_offsets,
                self.cls_keypoint_cfg['keypoint_types'], self.cls_transformer)
        if self.reg_keypoint_cfg['with_relation']:
            reg_feats = self._apply_relation(
                reg_feats, reg_feats, keypoint_scores, keypoint_offsets,
                self.reg_keypoint_cfg['keypoint_types'], self.reg_transformer)
        bbox_outs = self.bbox_head.forward_predictions(cls_feats, reg_feats)

        if self.training:
            return bbox_outs, keypoint_scores, keypoint_offsets
        else:
            return bbox_outs

    def _apply_relation(self, feats, keypoint_feats, keypoint_logits,
                        keypoint_offsets, keypoint_choices, transformer):
        # extract keypoints in each level
        keypoint_scores, keypoint_positions, keypoint_inds, locations = \
            self.keypoint_head.get_keypoints_multihead(
                keypoint_logits,
                keypoint_offsets,
                keypoint_choices,
                map_back=False,
                **self.keypoint_cfg)
        # extract keypoint features
        keypoint_features, keypoint_positions = \
            self.keypoint_head.get_keypoint_features(
                keypoint_feats,
                keypoint_scores,
                keypoint_positions,
                keypoint_inds,
                num_keypoint_head=len(keypoint_choices),
                **self.feature_selection_cfg)

        feat_sizes = [feat.size()[-2:] for feat in feats]
        query_features = [feat.permute(0, 2, 3, 1) for feat in feats]
        query_positions = [
            loc.reshape(*feat_size, -1)[None, ..., :2].expand(
                (query_features[0].size()[0], *feat_size, -1))
            for loc, feat_size in zip(locations, feat_sizes)
        ]
        if self.scale_position:
            scale_terms = [
                stride * self.scale_factor
                for stride in self.keypoint_head.strides
            ]
        else:
            scale_terms = [
                self.keypoint_head.strides[0] * self.scale_factor
                for _ in range(len(query_features))
            ]
        # forward with transformer
        query_features, _ = transformer(
            query_features,  # List[torch.Tensor(N,H,W,C)]
            query_positions,  # List[torch.Tensor(N,H,W,2)]
            keypoint_features,  # List[torch.Tensor(N,K,C)]
            keypoint_positions,  # List[torch.Tensor(N,K,2)]
            scale_terms)

        return [feat.permute(0, 3, 1, 2) for feat in query_features]

    def loss(self,
             bbox_outs,
             keypoint_scores,
             keypoint_offsets,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        keypoint_loss = self.keypoint_head.loss_multihead(
            keypoint_scores, keypoint_offsets, gt_bboxes, gt_labels, img_metas)
        bbox_loss = self.bbox_head.loss(*bbox_outs, gt_bboxes, gt_labels,
                                        img_metas, gt_bboxes_ignore)
        loss = dict()
        loss.update({'kpt_' + k: v for k, v in keypoint_loss.items()})
        loss.update({'bbox_' + k: v for k, v in bbox_loss.items()})

        return loss

    def get_bboxes(self, *args, **kwargs):
        return self.bbox_head.get_bboxes(*args, **kwargs)

    @property
    def num_classes(self):
        return self.bbox_head.num_classes

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test-time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        fusion_cfg = self.test_cfg.get('fusion_cfg', None)
        fusion_method = fusion_cfg.type if fusion_cfg else 'simple'
        assert fusion_method == 'soft_vote', (
            'BVR only supports soft_vote TTA now')

        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
