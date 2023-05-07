import copy
import datetime
import time

import numpy as np

from .api_wrappers import COCOeval


def calc_area_range_info(area_range_type):
    """Calculate area ranges and related information."""
    # use COCO setting as default
    area_labels = ['all', 'small', 'medium', 'large']
    scale_ranges = [[0, 1e5], [0, 32], [32, 96], [96, 1e5]]
    peak_ranges = [None] * 4
    relative_area = False

    if area_range_type == 'COCO':
        pass
    elif area_range_type == 'relative_scale_ap':
        relative_area = True
        area_labels = ['all']
        scale_ranges = [[0, 1]]
        peak_ranges = [None]
        inv_scale_thrs = np.power(2, np.arange(0, 10))[::-1]
        for inv_min, inv_max in zip(inv_scale_thrs[:-1], inv_scale_thrs[1:]):
            if inv_max == 256:
                area_labels.append(f'0_1/{inv_max}')
                scale_ranges.append([0, 1 / inv_max])
            else:
                area_labels.append(f'1/{inv_min}_1/{inv_max}')
                scale_ranges.append([1 / inv_min, 1 / inv_max])
            peak_ranges.append(None)
    elif area_range_type == 'absolute_scale_ap':
        scale_thrs = np.power(2, np.arange(2, 12))
        scale_thrs[0] = 0
        scale_thrs[-1] = 1e5
        for min_scale, max_scale in zip(scale_thrs[:-1], scale_thrs[1:]):
            area_labels.append(f'{min_scale:.0f}_{max_scale:.0f}')
            scale_ranges.append([min_scale, max_scale])
            peak_ranges.append(None)
    elif area_range_type == 'band_asap':
        scale_thrs = np.power(2, np.arange(1, 12))
        scale_thrs[0] = 1
        scale_thrs[-1] = 1e5
        for min_scale, peak_left, max_scale in zip(scale_thrs[:-2],
                                                   scale_thrs[1:-1],
                                                   scale_thrs[2:]):
            peak_right = 1e5 if max_scale == 1e5 else peak_left
            area_labels.append(
                f'{min_scale:.0f}_{peak_left:.0f}_{max_scale:.0f}')
            scale_ranges.append([min_scale, max_scale])
            peak_ranges.append([peak_left, peak_right])
    elif area_range_type == 'absolute_scale_ap_linear':
        scale_thrs = np.arange(0, 1024 + 32 + 1, 32)
        scale_thrs[-1] = 1e5
        for min_scale, max_scale in zip(scale_thrs[:-1], scale_thrs[1:]):
            area_labels.append(f'{min_scale:.0f}_{max_scale:.0f}')
            scale_ranges.append([min_scale, max_scale])
            peak_ranges.append(None)
    elif area_range_type == 'TinyPerson':
        area_labels = [
            'all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable'
        ]
        scale_ranges = [[1, 1e5], [1, 20], [1, 8], [8, 12], [12, 20], [20, 32],
                        [32, 1e5]]
        peak_ranges = [None] * 7
    else:
        raise NotImplementedError

    assert len(area_labels) == len(scale_ranges) == len(peak_ranges)
    scale_range_map = dict(zip(area_labels, scale_ranges))
    print('Scale ranges:', str(scale_range_map))
    peak_range_map = dict(zip(area_labels, peak_ranges))
    print('Peak ranges:', str(peak_range_map))

    return area_labels, scale_ranges, peak_ranges, relative_area


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
        area_labels, scale_ranges, peak_ranges, relative_area = \
            calc_area_range_info(area_range_type)
        self.params.areaRngLbl = area_labels
        self.params.areaRng = [[r[0]**2, r[1]**2] for r in scale_ranges]
        self.params.scale_ranges = scale_ranges
        self.params.peak_ranges = peak_ranges
        self.relative_area = relative_area
        for scale_range, peak_range in zip(scale_ranges, peak_ranges):
            if peak_range is None:
                assert scale_range[0] <= scale_range[1]
            else:
                assert scale_range[0] <= peak_range[0] <= peak_range[
                    1] <= scale_range[1]
                assert scale_range[0] > 0, \
                    'set positive scale_range to avoid -inf by log2'

    def evaluate(self):
        """Run per image evaluation on given images and store results (a list
        of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.
                  format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet, scale_range, peak_range)
            for catId in catIds for areaRng, scale_range, peak_range in zip(
                p.areaRng, p.scale_ranges, p.peak_ranges) for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def _calc_weight(self, scale, scale_range, peak_range=None):
        # for fast calculation
        if peak_range is None:
            if scale < scale_range[0] or scale > scale_range[1]:
                return 0.0
            else:
                return 1.0

        # trapezoidal filter in log space
        log_scale = np.log2(scale)
        trapezoid_x = np.array(
            (scale_range[0], peak_range[0], peak_range[1], scale_range[1]))
        a, b, c, d = np.log2(trapezoid_x)
        condlist = [
            log_scale < a,
            a <= log_scale < b,
            b <= log_scale <= c,
            c < log_scale <= d,
            log_scale > d,
        ]
        funclist = [
            0.0,
            lambda x: (x - a) / (b - a),
            1.0,
            lambda x: (x - d) / (c - d),
            0.0,
        ]
        return np.piecewise(log_scale, condlist, funclist)

    def evaluateImg(self,
                    imgId,
                    catId,
                    aRng,
                    maxDet,
                    scale_range,
                    peak_range=None):
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

        # divide area for thresholding with absolute/relative scale
        divisor = img_area if self.relative_area else 1
        for g in gt:
            area = g['area'] / divisor
            scale = area**0.5
            g['weight'] = self._calc_weight(scale, scale_range, peak_range)
            if g['ignore']:
                g['weight'] = 0.0
        for d in dt:
            area = d['area'] / divisor
            scale = area**0.5
            d['weight'] = self._calc_weight(scale, scale_range, peak_range)

        # sort gt highest weight first
        gtind = np.argsort([-g['weight'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        # sort dt highest score first
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
        gt_weights = np.array([g['weight'] for g in gt])
        dt_weights = np.array([d['weight'] for d in dt])
        dt_weights = np.repeat(dt_weights.reshape((1, len(dt))), T, 0)
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
                        if m > -1 and gt_weights[m] and not gt_weights[gind]:
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
                    dt_weights[tind, dind] = gt_weights[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'scale_range': scale_range,
            'peak_range': peak_range,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gt_weights': gt_weights,
            'dt_weights': dt_weights,
        }

    def accumulate(self, p=None):
        """Accumulate per image evaluation results and store the result in
        self.eval.

        :param p: input params for evaluation
        :return: None
        """
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dt_weights = np.concatenate(
                        [e['dt_weights'][:, 0:maxDet] for e in E],
                        axis=1)[:, inds]
                    gt_weights = np.concatenate([e['gt_weights'] for e in E])
                    npig = np.sum(gt_weights)
                    if npig == 0:
                        continue
                    tps = (dtm != 0) * dt_weights
                    fps = (dtm == 0) * dt_weights

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except IndexError:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100):
        """Compute and display a specific metric."""
        p = self.params
        iStr = (' {:<18} {} @[ IoU={:<9} | area={:>16s} | maxDets={:>4d} ]'
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
