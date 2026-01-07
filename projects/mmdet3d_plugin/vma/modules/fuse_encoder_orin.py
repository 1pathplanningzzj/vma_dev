import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils import Registry, build_from_cfg
from ..builder import FUSE_ENCODERS


# 实现简单融合编码器
@FUSE_ENCODERS.register_module()
class SimpleFusionEncoder(nn.Module):
    def __init__(self, in_channels=7, out_channels=3, conv_cfg=None, norm_cfg=None):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels,
            64,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            64,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
    
    def forward(self, img, view_img):
        x = torch.cat([img, view_img], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def build_fuse_encoder(cfg, default_args=None):
    """根据配置构建融合编码器"""
    return build_from_cfg(cfg, FUSE_ENCODERS, default_args)


if __name__ == "__main__":
    img=torch.rand([2,3,1000,1000])
    view_img=torch.rand([2,3,1000,1000])
    model=SimpleFusionEncoder(in_channels=6,out_channels=3)
    out=model(img,view_img)
    print(out.shape)