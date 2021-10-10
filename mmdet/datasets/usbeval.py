import numpy as np

from .api_wrappers import COCOeval


def calc_area_range_info(area_range_type):
    """Calculate area ranges and related information."""
    # use COCO setting as default
    area_ranges = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2],
                   [96**2, 1e5**2]]
    area_labels = ['all', 'small', 'medium', 'large']
    relative_area = False

    if area_range_type == 'COCO':
        pass
    elif area_range_type == 'relative_scale_ap':
        relative_area = True
        area_ranges = [[0**2, 1**2]]
        area_labels = ['all']
        inv_scale_thrs = np.power(2, np.arange(0, 10))[::-1]
        for inv_min, inv_max in zip(inv_scale_thrs[:-1], inv_scale_thrs[1:]):
            if inv_max == 256:
                area_ranges.append([0**2, 1 / inv_max**2])
                area_labels.append(f'0_1/{inv_max}')
            else:
                area_ranges.append([1 / inv_min**2, 1 / inv_max**2])
                area_labels.append(f'1/{inv_min}_1/{inv_max}')
    elif area_range_type == 'absolute_scale_ap':
        scale_thrs = np.power(2, np.arange(2, 12))
        scale_thrs[0] = 0
        scale_thrs[-1] = 1e5
        for min_scale, max_scale in zip(scale_thrs[:-1], scale_thrs[1:]):
            area_ranges.append([min_scale**2, max_scale**2])
            area_labels.append(f'{min_scale:.0f}_{max_scale:.0f}')
    elif area_range_type == 'absolute_scale_ap_linear':
        scale_thrs = np.arange(0, 1024 + 32 + 1, 32)
        scale_thrs[-1] = 1e5
        for min_scale, max_scale in zip(scale_thrs[:-1], scale_thrs[1:]):
            area_ranges.append([min_scale**2, max_scale**2])
            area_labels.append(f'{min_scale:.0f}_{max_scale:.0f}')
    elif area_range_type == 'TinyPerson':
        area_ranges = [[1**2, 1e5**2], [1**2, 20**2], [1**2, 8**2],
                       [8**2, 12**2], [12**2, 20**2], [20**2, 32**2],
                       [32**2, 1e5**2]]
        area_labels = [
            'all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable'
        ]
    else:
        raise NotImplementedError

    assert len(area_ranges) == len(area_labels)
    area_range_map = dict(zip(area_labels, area_ranges))
    print('Area ranges:', str(area_range_map))

    return area_ranges, area_labels, relative_area


class USBeval(COCOeval):

    def __init__(self,
                 cocoGt=None,
                 cocoDt=None,
                 iouType='segm',
                 area_range_type='COCO'):
        """Initialize CocoEval using coco APIs for gt and dt.

        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        area_ranges, area_labels, relative_area = calc_area_range_info(
            area_range_type)
        self.params.areaRng = area_ranges
        self.params.areaRngLbl = area_labels
        self.relative_area = relative_area

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        img_info = self.cocoGt.loadImgs([imgId])[0]
        img_area = img_info['width'] * img_info['height']

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if self.relative_area:
                area = g['area'] / img_area
            else:
                area = g['area']
            if g['ignore'] or (area < aRng[0] or area > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        if self.relative_area:
            a = np.array([
                d['area'] / img_area < aRng[0]
                or d['area'] / img_area > aRng[1] for d in dt
            ])
        else:
            a = np.array(
                [d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt])
        a = a.reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T,
                                                                      0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100):
        """Compute and display a specific metric."""
        p = self.params
        iStr = (' {:<18} {} @[ IoU={:<9} | area={:>11s} | maxDets={:>4d} ]'
                ' = {:0.3f}')
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    @staticmethod
    def _shorten_area_label(area_label):
        """Shorten area label like mmdet."""
        area_label_short = area_label
        shortening_map = {'small': 's', 'medium': 'm', 'large': 'l'}
        for long, short in shortening_map.items():
            area_label_short = area_label_short.replace(long, short)
        return area_label_short

    def _summarizeDets(self):
        """Compute and display summary metrics for detection."""
        max_dets = self.params.maxDets
        stats = {}
        # summarize AP
        for area_label in self.params.areaRngLbl:
            area_label_short = self._shorten_area_label(area_label)
            if area_label == 'all':
                stats['mAP'] = self._summarize(1)
                stats['mAP_50'] = self._summarize(
                    1, iouThr=.5, maxDets=max_dets[2])
                stats['mAP_75'] = self._summarize(
                    1, iouThr=.75, maxDets=max_dets[2])
            else:
                stats[f'mAP_{area_label_short}'] = self._summarize(
                    1, areaRng=area_label, maxDets=max_dets[2])
        # summarize AR
        for area_label in self.params.areaRngLbl:
            area_label_short = self._shorten_area_label(area_label)
            if area_label == 'all':
                for max_det in max_dets:
                    stats[f'AR@{max_det}'] = self._summarize(
                        0, maxDets=max_det)
            elif area_label in ['small', 'medium', 'large']:
                key = f'AR_{area_label_short}@{max_dets[2]}'
                stats[key] = self._summarize(
                    0, areaRng=area_label, maxDets=max_dets[2])
        return stats

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = self._summarizeDets
        elif iouType == 'keypoints':
            raise NotImplementedError
        self.stats = summarize()
