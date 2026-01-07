# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ================= copy meta and improve 11.07 last change ============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES

@LOSSES.register_module()
class GramLoss(nn.Module):
    """GramLoss：严格遵循DINOv3论文4.2-4.3节逻辑，修复bmm维度错误，实现局部窗口结构对齐"""
    def __init__(
        self,
        window_size=3,  
        img_level=True,  
        apply_norm=True,  
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss() # 保留以备不时之需，但主逻辑改用手动计算
        self.window_size = window_size  
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, mask=None, img_level=True, spatial_shape=None):
        """
        新增 mask 参数：
        mask (Tensor, optional): 形状为 (B, N) 或 (B, N, 1)。
                               用于指示哪些位置是有效的（LiDAR有响应的区域）。
        spatial_shape (tuple, optional): (H, W)，特征图的真实尺寸。如果不提供，默认为正方形。
        """
        # 1. 输入校验
        assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3, "输入必须为(B, N, dim)"
        B, N, dim = output_feats.shape
        
        if spatial_shape is not None:
            H, W = spatial_shape
            assert H * W == N, f"提供的尺寸 {H}x{W}={H*W} 与输入长度 {N} 不匹配"
        else:
            H = W = int(N ** 0.5)
            assert H * W == N, "输入特征长度不是完全平方数，且未提供 spatial_shape，这可能导致空间结构错乱"

        win_pixels = self.window_size ** 2 

        # 2. 特征重塑
        output_feat_map = output_feats.transpose(1, 2).reshape(B, dim, H, W)
        target_feat_map = target_feats.transpose(1, 2).reshape(B, dim, H, W)

        # 3. 局部窗口划分
        pad = (self.window_size - 1) // 2
        
        output_windows = F.unfold(
            F.pad(output_feat_map, (pad, pad, pad, pad)),
            kernel_size=self.window_size,
            stride=1
        ).reshape(B, dim, win_pixels, N)

        target_windows = F.unfold(
            F.pad(target_feat_map, (pad, pad, pad, pad)),
            kernel_size=self.window_size,
            stride=1
        ).reshape(B, dim, win_pixels, N)

        # 4. 维度调整 (B*N, win_pixels, dim)
        output_windows_3d = output_windows.permute(0, 3, 2, 1).reshape(B * N, win_pixels, dim)
        target_windows_3d = target_windows.permute(0, 3, 2, 1).reshape(B * N, win_pixels, dim)

        # 5. 计算 Gram 矩阵 (B*N, win_pixels, win_pixels)
        student_sim = torch.bmm(output_windows_3d, output_windows_3d.transpose(1, 2))
        target_sim = torch.bmm(target_windows_3d, target_windows_3d.transpose(1, 2))

        # 6. 归一化
        student_sim = student_sim / dim
        target_sim = target_sim / dim

        # 7. 维度恢复 (B, N, win_pixels, win_pixels)
        student_sim = student_sim.reshape(B, N, win_pixels, win_pixels)
        target_sim = target_sim.reshape(B, N, win_pixels, win_pixels)

        # 8. 负相似度过滤
        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0
        elif self.remove_only_teacher_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        # ==================== 9. 核心修改：带 Mask 的 Loss 计算 ====================
        
        # 计算逐元素的平方误差 (Square Error Map)
        # 形状: (B, N, win_pixels, win_pixels)
        loss_map = (student_sim - target_sim) ** 2

        if mask is not None:
            # 调整 mask 维度以适配广播
            # 假设 mask 输入是 (B, N) 或 (B, N, 1)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1).unsqueeze(-1) # (B, N, 1, 1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(-1) # (B, N, 1, 1)
            
            # 加权求和
            weighted_loss = (loss_map * mask).sum()
            
            # 归一化：除以 mask 的总权重 * 矩阵元素数量
            # 添加 1e-6 防止除以 0
            normalizer = mask.sum() * (win_pixels ** 2)
            loss = weighted_loss / (normalizer + 1e-6)
            
            return loss
        else:
            # 如果没有 mask，退化为普通 MSE
            return loss_map.mean()
