from typing import List, Tuple

import numpy as np
import torch

from ..builder import BBOX_ASSIGNERS
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class PointKptAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, gaussian_iou=0.7):
        self.gaussian_iou = gaussian_iou

    def assign(self,
               points: torch.Tensor,
               num_points_list: List[int],
               gt_points: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor = None,
               num_classes: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign gt to points.

        Args:
            points (torch.Tensor): Points for all level with shape
                (num_points, 3) in [x,y,stride] format.
            num_points_list (List[int]): point number for each level.
            gt_points (torch.Tensor): Ground truth points for single image with
                shape (num_gts, 2) in [x, y] format.
            gt_bboxes (torch.Tensor): Ground truth bboxes of single image,
                each has shape (num_gt, 4).
            gt_labels (torch.Tensor): Ground truth labels of single image,
                each has shape (num_gt,).
        Returns:
            Tuple[torch.Tensor,torch.Tensor]: offset targets [num_points,2]
                and score_targets [num_points,num_gts]
        """
        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0 or num_points == 0:
            return points.new_zeros(num_points, 0), \
                   points.new_zeros(num_points, 0)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        gt_bboxes_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_bboxes_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        radius = gaussian_radius_torch((gt_bboxes_h, gt_bboxes_w),
                                       self.gaussian_iou)
        diameter = 2 * radius + 1
        sigma = diameter / 6
        sigma_square = sigma[None, :]**2
        # compute distance
        distance = torch.pow(points_xy[:, None, :] - gt_points[None, ...],
                             2).sum(dim=2)

        point_range = np.cumsum([0] + num_points_list)
        score_targets = points.new_zeros(num_points, num_gts)
        # offset_targets = points.new_zeros(num_points, 2)
        pos_mask = points.new_zeros(num_points, )

        for i, _ in enumerate(range(lvl_min, lvl_max + 1)):
            distance_per_lvl = distance[point_range[i]:point_range[i + 1]]
            _, min_distance_inds = distance_per_lvl.min(dim=0)  # [num_gts]
            pos_points_inds = point_range[i] + min_distance_inds
            selected_points = points[pos_points_inds]  # num_gt,3
            # offset_targets[pos_points_inds, :] = (
            #     gt_points - selected_points[:, :2]) / selected_points[:, 2:]
            offset = points_xy[point_range[i]:point_range[i + 1],
                               None, :2] - selected_points[None, :, :2]
            score_targets[point_range[i]:point_range[i + 1]] = torch.exp(
                -0.5 * torch.pow(offset, 2).sum(dim=-1) / sigma_square)
            pos_mask[pos_points_inds] = 1

        score_target_max, max_inds = score_targets.max(dim=1)
        offset_targets = points.new_zeros(num_points, 2)
        valid_points = points[pos_mask == 1]
        valid_point_gts = gt_points[max_inds][pos_mask == 1]
        offset_targets[pos_mask == 1] = (valid_point_gts[:, :2] -
                                         valid_points[:, :2]) / (
                                             valid_points[:, 2:])

        if num_classes is None:
            score_target = score_target_max
        else:
            score_target = points.new_zeros(points.size(0), num_classes)
            score_target[
                torch.arange(points.size(0)),
                gt_labels.index_select(0, max_inds)] = score_target_max

        return offset_targets, score_target, pos_mask


def gaussian_radius_torch(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    r = torch.stack([r1, r2, r3], dim=1)
    return torch.min(r, dim=1)[0]
