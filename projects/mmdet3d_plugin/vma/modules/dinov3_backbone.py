import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from mmdet3d.models.builder import BACKBONES

@BACKBONES.register_module()
class DINOv3Backbone(nn.Module):
    def __init__(
        self,
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        out_channels=256,
        frozen_stages=8,  # 冻结前8层，只微调后几层（ViT共12层）
        norm_eval=True,
        **kwargs
    ):
        super().__init__()
        # 加载DINOv3预训练模型和处理器
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.dinov3 = AutoModel.from_pretrained(model_id)
        self.out_channels = out_channels
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # 特征投影层：将DINOv3的768通道 → Neck需要的256通道
        self.proj = nn.Sequential(
            nn.Conv2d(768, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels),  # 与原Neck的GN配置一致
            nn.ReLU(inplace=True)
        )

        # 冻结指定层数
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # 冻结嵌入层和前frozen_stages层encoder
            for param in self.dinov3.embeddings.parameters():
                param.requires_grad = False
            for i in range(self.frozen_stages):
                if i < len(self.dinov3.encoder.layer):
                    for param in self.dinov3.encoder.layer[i].parameters():
                        param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: 输入图像张量 (B, 3, H, W)，对应原配置的 lidar_map/view_map
        Returns:
            多尺度特征列表（适配原Neck的输入格式）
        """
        B, C, H, W = x.shape

        # DINOv3要求输入为 (B, H, W, C)，且已归一化，这里直接适配
        x = x.permute(0, 2, 3, 1)  # (B, H, W, 3)

        # 前向传播：获取最后一层encoder的输出（CLS token + 像素token）
        outputs = self.dinov3(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state  # (B, H*W + 1, 768)

        # 去除CLS token，保留像素token → 重塑为特征图
        pixel_feat = last_hidden_state[:, 1:, :]  # (B, H*W, 768)
        pixel_feat = pixel_feat.permute(0, 2, 1).reshape(B, 768, H, W)  # (B, 768, H, W)

        # 投影到256通道
        feat = self.proj(pixel_feat)  # (B, 256, H, W)

        # 适配原Neck的多尺度输入格式（原ResNet输出3个尺度，这里用1个尺度重复3次，或通过下采样生成多尺度）
        # 简化方案：输出3个相同尺度的特征（后续可优化为多尺度）
        return [feat, feat, feat]

    def train(self, mode=True):
        super().train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.GroupNorm):
                    m.eval()