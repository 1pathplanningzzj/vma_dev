import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

ATTR_HEAD = Registry('Attr_Head')

def build_attr_head(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, ATTR_HEAD, default_args)

# 创建融合编码器注册器
FUSE_ENCODERS = Registry('fuse_encoder')
FUSE_NECKS = Registry('fuse_neck')

TRANSFORMERS  = Registry('vma_transformer')

def build_vma_transformer(cfg, default_args=None):
    return build_from_cfg(cfg, TRANSFORMERS , default_args)


