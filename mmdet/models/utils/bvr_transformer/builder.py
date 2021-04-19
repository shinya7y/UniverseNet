from mmcv.utils import Registry, build_from_cfg

BVR_ATTENTION = Registry('Attention_Layer')
BVR_POSITIONAL_ENCODING = Registry('Positional_Encoding')
BVR_TRANSFORMER = Registry('BVR_Transformer')


def build_attention_layer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, BVR_ATTENTION, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, BVR_POSITIONAL_ENCODING, default_args)


def build_transformer(cfg, default_args=None):
    return build_from_cfg(cfg, BVR_TRANSFORMER, default_args)
