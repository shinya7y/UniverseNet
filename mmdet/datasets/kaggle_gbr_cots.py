# Copyright (c) OpenMMLab. All rights reserved.
# Code for GBR competition metric is modified from Camaro's notebook
# https://www.kaggle.com/bamps53/competition-metric-implementation
import numpy as np
from mmcv.utils import print_log

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from .builder import DATASETS
from .coco import CocoDataset


def f_beta(tp, fp, fn, beta=2):
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)


def calc_tpfpfn(gt_bboxes, pred_bboxes, iou_thr=0.5):
    """Calculate tp, fp, fn.

    gt_bboxes: (N, 4) np.array in xyxy format
    pred_bboxes: (N, 5) np.array in xyxy+conf format
    """
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, 0
        return tps, fps, fns
    elif len(gt_bboxes) == 0:
        tps, fps, fns = 0, len(pred_bboxes), 0
        return tps, fps, fns
    elif len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, len(gt_bboxes)
        return tps, fps, fns

    # sort by conf
    pred_bboxes = pred_bboxes[pred_bboxes[:, 4].argsort()[::-1]]
    gt_bboxes = gt_bboxes.copy()

    tp = 0
    fp = 0
    for k, pred_bbox in enumerate(pred_bboxes):
        ious = bbox_overlaps(gt_bboxes, pred_bbox[None, :4])
        max_iou = ious.max()
        if max_iou > iou_thr:
            tp += 1
            gt_bboxes = np.delete(gt_bboxes, ious.argmax(), axis=0)
        else:
            fp += 1
        if len(gt_bboxes) == 0:
            fp += len(pred_bboxes) - (k + 1)
            break
    fn = len(gt_bboxes)

    return tp, fp, fn


def calc_f2_score(gt_bboxes_list,
                  pred_bboxes_list,
                  iou_thr=0.5,
                  verbose=False):
    """Calculate F2-score.

    gt_bboxes_list: list of (N, 4) np.array in xyxy format
    pred_bboxes_list: list of (N, 5) np.array in xyxy+conf format
    """
    tps, fps, fns = 0, 0, 0
    for gt_bboxes, pred_bboxes in zip(gt_bboxes_list, pred_bboxes_list):
        tp, fp, fn = calc_tpfpfn(gt_bboxes, pred_bboxes, iou_thr)
        tps += tp
        fps += fp
        fns += fn
        if verbose:
            num_gt = len(gt_bboxes)
            num_pred = len(pred_bboxes)
            print(f'num_gt:{num_gt:<3} num_pred:{num_pred:<3}'
                  f' tp:{tp:<3} fp:{fp:<3} fn:{fn:<3}')
    f2 = f_beta(tps, fps, fns, beta=2)

    return f2


@DATASETS.register_module()
class GBRCOTSDataset(CocoDataset):
    """Dataset of crown-of-thorns starfish on Great Barrier Reef.

    https://www.kaggle.com/c/tensorflow-great-barrier-reef
    """

    CLASSES = ('cots', )

    PALETTE = [(255, 111, 0)]

    def prepare_gt_bboxes_list(self):
        gt_bboxes_list = []
        for img_info in self.data_infos:
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes = np.zeros([0, 4])
            else:
                gt_bboxes = [ann['bbox'] for ann in ann_info]
                gt_bboxes = np.array(gt_bboxes)
                gt_bboxes[:, 2:] += gt_bboxes[:, :2]  # xywh2xyxy
            gt_bboxes_list.append(gt_bboxes)

        return gt_bboxes_list

    def evaluate_mean_f2(self,
                         results,
                         iou_thrs=[0.5],
                         score_thr=0.,
                         logger=None):
        """Evaluate mean F2-score."""
        num_classes_of_results = len(results[0])
        if num_classes_of_results != 1:
            raise NotImplementedError

        gt_bboxes_list = self.prepare_gt_bboxes_list()
        cots_results = [result[0] for result in results]  # cots class only
        pred_bboxes_list = [det[det[:, 4] > score_thr] for det in cots_results]

        f2_scores = []
        for iou_thr in iou_thrs:
            f2_score = calc_f2_score(
                gt_bboxes_list, pred_bboxes_list, iou_thr=iou_thr)
            f2_scores.append(f2_score)
            log_msg = f'f2 @[ IoU={iou_thr:.2f}      | score>{score_thr:.2f} ] = {f2_score:.4f}'  # noqa
            print_log(log_msg, logger=logger)

        mean_f2_score = np.mean(f2_scores)
        log_msg = f'f2 @[ IoU={iou_thrs[0]:.2f}:{iou_thrs[-1]:.2f} | score>{score_thr:.2f} ] = {mean_f2_score:.4f}'  # noqa
        print_log(log_msg, logger=logger)

        return mean_f2_score

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(1, 10, 100),
                 iou_thrs=None,
                 score_thrs=None,
                 metric_items=None):
        """Evaluate COCO metrics and mean F2-score."""
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .3, .8, int(np.round((.8 - .3) / .05)) + 1, endpoint=True)
        if score_thrs is None:
            score_thrs = np.linspace(
                .05, .95, int(np.round((.95 - .05) / .05)) + 1, endpoint=True)

        # evaluate COCO metrics
        eval_results = super().evaluate(
            results,
            metric=metric,
            logger=logger,
            jsonfile_prefix=jsonfile_prefix,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items)

        # evaluate mean F2-score
        for score_thr in score_thrs:
            mean_f2 = self.evaluate_mean_f2(
                results, iou_thrs=iou_thrs, score_thr=score_thr, logger=logger)
            eval_results[f'mean_f2_{score_thr:.2f}'] = float(f'{mean_f2:.4f}')

        return eval_results
