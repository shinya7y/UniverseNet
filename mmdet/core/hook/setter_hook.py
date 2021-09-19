from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EpochSetterHook(Hook):
    """Epoch setter hook.

    This hook can be used to notify current epoch to model.
    """

    def __init__(self, target_vars):
        self.target_vars = target_vars

    def before_epoch(self, runner):
        for var in self.target_vars:
            exec(f'{var} = runner.epoch')


@HOOKS.register_module()
class IterSetterHook(Hook):
    """Iteration setter hook.

    This hook can be used to notify current iteration to model.
    """

    def __init__(self, target_vars):
        self.target_vars = target_vars

    def before_iter(self, runner):
        for var in self.target_vars:
            exec(f'{var} = runner.iter')
