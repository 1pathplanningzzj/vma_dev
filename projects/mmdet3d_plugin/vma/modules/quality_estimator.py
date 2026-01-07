import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule  # 复用MMCV的ConvModule，含归一化和激活


class ViewQualityEstimator(nn.Module):
    """
    评估view_map的场景有效性（雨天/夜间等场景分数高，晴天强光分数低）
    输入：view_map图像 [B, 3, H, W]（RGB格式）
    输出：质量分数 [B, 1]（0~1，分数越高表示view_map越有效）
    """
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        # 轻量级特征提取：捕捉view_map的纹理、亮度、对比度等特征（判断场景有效性）
        self.feat_extractor = nn.Sequential(
            # 第1卷积：3→hidden_dim//2，下采样1倍
            ConvModule(
                in_channels=in_channels,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN'),  # 批量归一化，稳定训练
                act_cfg=dict(type='ReLU')  # ReLU激活
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样2倍，减少计算量
            
            # 第2卷积：hidden_dim//2→hidden_dim，下采样1倍
            ConvModule(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 再下采样2倍
            
            # 全局平均池化：将任意H×W的特征图转为1×1（与输入尺寸无关）
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # 展平为向量 [B, hidden_dim]
        )
        
        # 质量分数预测：将特征向量转为0~1的分数
        self.score_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 降维
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 最终输出1个分数
            nn.Sigmoid()  # 激活到0~1范围
        )

    def forward(self, view_map):
        """前向传播：view_map → 特征提取 → 质量分数"""
        # 1. 特征提取：[B,3,H,W] → [B, hidden_dim]
        feat = self.feat_extractor(view_map)
        # 2. 分数预测：[B, hidden_dim] → [B,1]
        quality_score = self.score_predictor(feat)
        return quality_score