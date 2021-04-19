from inspect import signature

import numpy as np
import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from mmdet.models.utils.bvr_transformer import SimpleBVR_Transformer
from mmdet.models.utils.bvr_utils import assign_required_method
from ..builder import HEADS, build_head
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

INF = 1e8


@HEADS.register_module()
class BVRHead(BaseDenseHead, BBoxTestMixin):

    def __init__(
        self,
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
        test_cfg: dict = None,
    ):
        """BVRHead for single stage detector.

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
        super().__init__()
        if keypoint_cfg is None:
            keypoint_cfg = dict(
                max_keypoint_num=20,
                keypoint_score_thr=0.0,
                fuse_multi_level=False,
                max_fuse_keypoint_num=50,
            )
        if feature_selection_cfg is None:
            feature_selection_cfg = dict()
        if reg_keypoint_cfg is None:
            reg_keypoint_cfg = dict(
                keypoint_types=['top_left_corner', 'bottom_right_corner'],
                with_key_score=False,
                with_relation=False,
            )
        if cls_keypoint_cfg is None:
            cls_keypoint_cfg = dict(
                keypoint_types=['center'],
                with_key_score=False,
                with_relation=False,
            )
        if train_cfg is not None:
            bbox_head_cfg.update(train_cfg=train_cfg.bbox)
        if test_cfg is not None:
            bbox_head_cfg.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head_cfg)
        if assign_required_method(self.bbox_head) != 1:
            raise RuntimeError('BBox Head is not well modified.')
        if train_cfg is not None:
            keypoint_head_cfg.update(train_cfg=train_cfg.keypoint)
        self.keypoint_head = build_head(keypoint_head_cfg)

        self.keypoint_cfg = keypoint_cfg
        self.feature_selection_cfg = feature_selection_cfg
        self.cls_keypoint_cfg = cls_keypoint_cfg
        self.reg_keypoint_cfg = reg_keypoint_cfg
        self.with_relation = (
            cls_keypoint_cfg['with_relation']
            or reg_keypoint_cfg['with_relation'])
        self.keypoint_pos = ['input', 'cls', 'reg'].index(keypoint_pos)
        # build transformer
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
        print(self)

    def build_transformer(self):
        num_heads, pos_cfg = self.num_heads, self.pos_cfg
        if self.cls_keypoint_cfg['with_relation']:
            self.cls_transformer = SimpleBVR_Transformer(
                2,
                self.bbox_head.feat_channels,
                num_heads,
                len(self.cls_keypoint_cfg['keypoint_types']),
                with_relative_positional_encoding=self.
                with_relative_positional_encoding,
                with_appearance_relation=self.with_appearance_relation,
                positional_cfg=pos_cfg,
                shared_positional_encoding=self.
                shared_positional_encoding_inter,
            )
        if self.reg_keypoint_cfg['with_relation']:
            self.reg_transformer = SimpleBVR_Transformer(
                2,
                self.bbox_head.feat_channels,
                num_heads,
                len(self.reg_keypoint_cfg['keypoint_types']),
                positional_cfg=pos_cfg,
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

        if self.with_relation:
            if self.cls_keypoint_cfg['with_relation']:
                cls_feats = self._apply_relation(
                    cls_feats,
                    cls_feats,
                    keypoint_scores,
                    keypoint_offsets,
                    self.cls_keypoint_cfg['keypoint_types'],
                    self.cls_transformer,
                )
            if self.reg_keypoint_cfg['with_relation']:
                reg_feats = self._apply_relation(
                    reg_feats,
                    reg_feats,
                    keypoint_scores,
                    keypoint_offsets,
                    self.reg_keypoint_cfg['keypoint_types'],
                    self.reg_transformer,
                )
        bbox_outs = self.bbox_head.forward_predictions(cls_feats, reg_feats)

        if self.training:
            return bbox_outs, keypoint_scores, keypoint_offsets
        else:
            return bbox_outs

    def _apply_relation(self, feats, keypoint_feats, keypoint_logits,
                        keypoint_offsets, keypoint_choices, transformer):
        # extract keypoints in each level
        (
            keypoint_scores,
            keypoint_positions,
            keypoint_inds,
            locations,
        ) = self.keypoint_head.get_keypoints_multihead(
            keypoint_logits,
            keypoint_offsets,
            keypoint_choices,
            map_back=False,
            **self.keypoint_cfg)
        # extract keypoint features
        (
            keypoint_features,
            keypoint_positions,
        ) = self.keypoint_head.get_keypoint_features(
            keypoint_feats,
            keypoint_scores,
            keypoint_positions,
            keypoint_inds,
            num_keypoint_head=len(keypoint_choices),
            **self.feature_selection_cfg)

        feat_sizes = [feat.size()[-2:] for feat in feats]
        query_features = [feat.permute(0, 2, 3, 1) for feat in feats]
        query_positions = [
            loc.reshape(*feat_size,
                        -1)[None, ..., :2].expand(query_features[0].size()[0],
                                                  *feat_size, -1)
            for loc, feat_size in zip(locations, feat_sizes)
        ]
        # forward with transformer
        query_features, _ = transformer(
            query_features,  # List[torch.Tensor(N,H,W,C)]
            query_positions,  # List[torch.Tensor(N,H,W,2)]
            keypoint_features,  # List[torch.Tensor(N,K,C)]
            keypoint_positions,  # List[torch.Tensor(N,K,2)]
            [
                self.keypoint_head.strides[0] * self.scale_factor
                for _ in range(len(query_features))
            ] if not self.scale_position else [
                stride * self.scale_factor
                for stride in self.keypoint_head.strides
            ],
        )
        return [feat.permute(0, 3, 1, 2) for feat in query_features]

    def loss(
        self,
        bbox_outs,
        keypoint_scores,
        keypoint_offsets,
        gt_bboxes,
        gt_labels,
        img_metas,
        gt_bboxes_ignore=None,
    ):

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
        if self.test_cfg.get('method', 'simple') == 'simple':
            return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
        else:
            return self.aug_test_vote(feats, img_metas, rescale=rescale)

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation.

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
        # check with_nms argument
        gb_sig = signature(self.bbox_head.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self.bbox_head._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_factors = []  # score_factors for NMS
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            # bbox_outputs of some detectors (e.g., ATSS, FCOS, YOLOv3)
            # contains additional element to adjust scores before NMS
            if len(bbox_outputs) >= 3:
                aug_factors.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_factors)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels, self.num_classes)

        return bbox_results

    def aug_test_vote(self, feats, img_metas, rescale=False):

        # check with_nms argument
        gb_sig = signature(self.bbox_head.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self.bbox_head._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'
        aug_bboxes = []
        aug_labels = []
        for i, (x, img_meta) in enumerate(zip(feats, img_metas)):
            # only one image in the batch
            # TODO more flexible
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, True)
            det_bboxes, det_labels = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            keeped = self.remove_boxes(det_bboxes,
                                       self.test_cfg.scale_ranges[i // 2][0],
                                       self.test_cfg.scale_ranges[i // 2][1])
            det_bboxes, det_labels = det_bboxes[keeped, :], det_labels[keeped]
            aug_bboxes.append(det_bboxes)
            aug_labels.append(det_labels)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_labels = self.merge_aug_vote_results(
            aug_bboxes, aug_labels, img_metas)

        det_bboxes = []
        det_labels = []
        for j in range(80):
            inds = (merged_labels == j).nonzero().squeeze(1)

            scores_j = merged_bboxes[inds, 4]
            bboxes_j = merged_bboxes[inds, :4].view(-1, 4)
            bboxes_j, scores_j = self.bboxes_vote(bboxes_j, scores_j)

            if len(bboxes_j) > 0:
                det_bboxes.append(
                    torch.cat([bboxes_j, scores_j[:, None]], dim=1))
                det_labels.append(
                    torch.full((bboxes_j.shape[0], ),
                               j,
                               dtype=torch.int64,
                               device=scores_j.device))

        # select
        if len(det_bboxes) > 0:
            det_bboxes = torch.cat(det_bboxes, dim=0)
            det_labels = torch.cat(det_labels)
        else:
            det_bboxes = merged_bboxes.new_zeros((0, 5))
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)

        if det_bboxes.shape[0] > 1000 > 0:
            cls_scores = det_bboxes[:, 4]
            image_thresh, _ = torch.kthvalue(cls_scores.cpu(),
                                             det_bboxes.shape[0] - 1000 + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        # pdb.set_trace()
        return bbox_results

    def merge_aug_vote_results(self, aug_bboxes, aug_labels, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).
            rcnn_test_cfg (dict): rcnn test config.
        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes[:, :4] = bbox_mapping_back(bboxes[:, :4], img_shape,
                                              scale_factor, flip)
            if bboxes.size()[1] != 5:
                assert bboxes.size()[0] == 0
                bboxes = bboxes.new_zeros(0, 5)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_labels is None:
            return bboxes
        else:
            labels = torch.cat(aug_labels, dim=0)
            return bboxes, labels

    def remove_boxes(self, boxes, min_scale, max_scale):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = torch.nonzero(
            (areas >= min_scale * min_scale) &
            (areas <= max_scale * max_scale),
            as_tuple=False).squeeze(1)

        return keep

    def bboxes_vote(self, boxes, scores, vote_thresh=0.66):
        eps = 1e-6

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy().reshape(-1, 1)
        det = np.concatenate((boxes, scores), axis=1)
        if det.shape[0] <= 1:
            return np.zeros((0, 5)), np.zeros((0, 1))
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        dets = []
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = area[0] + area[:] - inter
            union = np.maximum(union, eps)
            o = inter / union
            o[0] = 1

            # get needed merge det and delete these  det
            merge_index = np.where(o >= vote_thresh)[0]
            det_accu = det[merge_index, :]
            det_accu_iou = o[merge_index]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                try:
                    dets = np.row_stack((dets, det_accu))
                except ValueError:
                    dets = det_accu
                continue
            else:
                soft_det_accu = det_accu.copy()
                soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
                soft_index = np.where(soft_det_accu[:, 4] >= 0.05)[0]
                soft_det_accu = soft_det_accu[soft_index, :]

                det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                    det_accu[:, -1:], (1, 4))
                max_score = np.max(det_accu[:, 4])
                det_accu_sum = np.zeros((1, 5))
                det_accu_sum[:, 0:4] = np.sum(
                    det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
                det_accu_sum[:, 4] = max_score

                if soft_det_accu.shape[0] > 0:
                    det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

                try:
                    dets = np.row_stack((dets, det_accu_sum))
                except ValueError:
                    dets = det_accu_sum

        order = dets[:, 4].ravel().argsort()[::-1]
        dets = dets[order, :]

        boxes = torch.from_numpy(dets[:, :4]).float().cuda()
        scores = torch.from_numpy(dets[:, 4]).float().cuda()

        return boxes, scores
