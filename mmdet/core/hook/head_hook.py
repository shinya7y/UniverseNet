from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class HeadHook(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        runner.model.module.bbox_head.epoch = runner.epoch
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
