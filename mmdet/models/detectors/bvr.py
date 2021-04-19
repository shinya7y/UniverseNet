from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class BVR(SingleStageDetector):
    """RelationNet++ (BVR).

    https://arxiv.org/abs/2010.15831
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BVR, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
