import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.cnn.bricks.transformer import build_attention


@TRANSFORMER_LAYER.register_module()
class VMADetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,** kwargs):
        # 关键修改：用 cross_attn 代替 cross_attn_view 和 cross_attn_lidar
        operation_order = (
            'cross_attn', 'norm',  # 第1个cross_attn：对应view分支
            'self_attn', 'norm',   # 自注意力
            'cross_attn', 'norm',  # 第2个cross_attn：对应lidar分支
            'ffn', 'norm'          # FFN
        )
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        # 验证注意力配置数量为3（2个cross_attn + 1个self_attn）
        assert len(attn_cfgs) == 3, f"需3个注意力配置，实际为{len(attn_cfgs)}"
    def build_attention(self, cfg, index):
        """构建3种注意力模块（按index区分类型）"""
        if index == 0:
            # 0: view交叉注意力（对应operation_order的cross_attn_view）
            return build_attention(cfg)
        elif index == 1:
            # 1: 自注意力（对应operation_order的self_attn）
            return build_attention(cfg)
        elif index == 2:
            # 2: lidar交叉注意力（对应operation_order的cross_attn_lidar）
            return build_attention(cfg)
        else:
            raise ValueError(f"注意力索引{index}超出范围（0-2）")