from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class TOOD(SingleStageDetector):
    """Implementation of `TOOD <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TOOD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
