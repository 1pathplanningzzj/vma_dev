import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn import ConvModule, build_norm_layer
from torchvision.ops import DeformConv2d
import sys
from pathlib import Path

# 项目路径配置（根据实际情况调整）
sys.path.append(str(Path(__file__).resolve().parents[4]))
from projects.mmdet3d_plugin.vma.builder import FUSE_NECKS

@FUSE_NECKS.register_module()
class ViewFeatureFusion(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], offset_channels=64):
        """
        Args:
            in_channels: 各阶段输入通道数 [stage1, stage2, stage3]
            offset_channels: 偏移估计器的中间通道数
        """
        super().__init__()
        
        # 可变形卷积参数组
        self.deform_groups = 4  # 分组数可调整
        
        # 每个stage的偏移估计器和可变形卷积
        self.offset_estimators = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        
        for ch in in_channels:
            # 偏移量估计网络 (输入: img+view_img拼接)
            self.offset_estimators.append(
                nn.Sequential(
                    nn.Conv2d(ch * 2, offset_channels, 3, padding=1),
                    nn.GroupNorm(8, offset_channels),  # 添加归一化
                    nn.ReLU(),
                    nn.Conv2d(offset_channels, self.deform_groups * 2 * 3 * 3, 3, padding=1),
                    nn.Tanh()  # 限制偏移范围
                ))
            
            # 可变形卷积 (处理view_img)
            deform_conv = DeformConv2d(
                ch, ch, 
                kernel_size=3,
                padding=1,
                groups=self.deform_groups
            )
            self.deform_convs.append(deform_conv)
            
        # 最终融合的1x1卷积 (可选)
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, 1) for ch in in_channels
        ])
        # 确保所有参数参与计算

    def forward(self, img_feats, view_feats):
        """
        Args:
            img_feats:   List[Tensor], ResNet提取的img特征 [stage1, stage2, stage3]
            view_feats:  List[Tensor], ResNet提取的view_img特征 (与img_feats同形状)
        Returns:
            fused_feats: List[Tensor], 融合后的特征
        """
        fused_feats = []
        
        for i, (img, view) in enumerate(zip(img_feats, view_feats)):
            
            # 偏移量计算
            offset = self.offset_estimators[i](torch.cat([img, view], dim=1))
            offset = offset.clamp(-3, 3)
            
            # 2. 对view_feat做可变形卷积
            deformed_view = self.deform_convs[i](view, offset)
            
            # 3. 与原始img特征融合 (相加或concat后1x1卷积)
            fused = torch.cat([img, deformed_view], dim=1)
            fused = self.fusion_convs[i](fused)
            
            fused_feats.append(fused)
            
        return fused_feats
# class CrossAttnBlock(nn.Module):
#     def __init__(self, attn_embed_dims, num_heads, attn_dropout, norm_cfg):
#         super().__init__()
#          # 将输入特征 线性映射到 Q K V 
#         self.q_proj = nn.Linear(attn_embed_dims, attn_embed_dims)
#         self.k_proj = nn.Linear(attn_embed_dims, attn_embed_dims)
#         self.v_proj = nn.Linear(attn_embed_dims, attn_embed_dims) #　y = x · W + b　　　Ｗ　可学习矩阵　［out_feats, in_feats］　
#         #  线性变换将输入特征转换为适合注意力计算的表示
#         #  多头注意力  
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=attn_embed_dims, num_heads=num_heads, dropout=attn_dropout, batch_first=True
#         )
#         #归一化　用于残差连接　
#         self.norm1 = build_norm_layer(norm_cfg, attn_embed_dims)[1] # LayerNorm   归一化 加速收敛 
#         self.norm2 = build_norm_layer(norm_cfg, attn_embed_dims)[1]
#        # FFN 前向   
#         self.ffn = nn.Sequential(
#             nn.Linear(attn_embed_dims, attn_embed_dims * 2),
#             nn.ReLU(),
#             nn.Linear(attn_embed_dims * 2, attn_embed_dims)
#         ) # 扩大 特征维度学习更多的信息 用RELU来拟合非线性 

#     def forward(self, q_seq, kv_seq):
#         """交叉注意力：q_seq（主序列）通过kv_seq（辅助序列）增强"""
#         q = self.q_proj(q_seq)
#         k = self.k_proj(kv_seq)
#         v = self.v_proj(kv_seq)
        
#         attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
#         q_seq = self.norm1(attn_out + q_seq)  # 残差1
        
#         ffn_out = self.ffn(q_seq)
#         q_seq = self.norm2(ffn_out + q_seq)  # 残差2
#         return q_seq
#         # q_seq 既保留了主模态的原始优势，又融入了辅助模态的补充信息（视觉的语义细节），实现了特征增强。

# class SequentialMultiInput(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         self.modules_list = nn.ModuleList(args)

#     def forward(self, x, y):
#         for module in self.modules_list:
#             x = module(x, y)
#         return x


# @FUSE_NECKS.register_module()
# class CrossAttnFeatEnhancer(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  patch_sizes=[[16,16], [8,8], [4,4]],
#                  num_attn_layers=3,
#                  attn_embed_dims=256,
#                  num_heads=8,
#                  attn_dropout=0.1,
#                  norm_cfg=dict(type='LN'),
#                  conv_norm_cfg=dict(type='BN'),
#                  conv_cfg=None,
#                  act_cfg=dict(type='ReLU')):
#         super().__init__()
#         self.in_channels = in_channels
#         self.patch_sizes = patch_sizes
#         self.num_attn_layers = num_attn_layers
#         self.attn_embed_dims = attn_embed_dims
#         self.num_heads = num_heads
        
#         # 移除DCN相关初始化（仅保留交叉注意力）
#         assert len(in_channels) == len(patch_sizes), "通道数与patch尺寸数必须一致"
#         assert attn_embed_dims % num_heads == 0, "attn_embed_dims需被num_heads整除"

#         self.stage_modules = nn.ModuleList()
#         for idx, (in_dim, p_size) in enumerate(zip(in_channels, patch_sizes)):
#             p_h, p_w = p_size
#             stage_module = nn.ModuleDict() # 处理 512 1024 2048 这三个维度的特征

#             # Patch维度投影（适配注意力输入）
#             patch_dim = in_dim * p_h * p_w
#             stage_module['patch_proj'] = ConvModule(
#                 in_channels=patch_dim,
#                 out_channels=attn_embed_dims,
#                 kernel_size=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=None,
#                 act_cfg=None
#             )

#             # 多层交叉注意力
#             stage_module['attn_blocks'] = SequentialMultiInput(
#                 *[CrossAttnBlock(
#                     attn_embed_dims=attn_embed_dims,
#                     num_heads=num_heads,
#                     attn_dropout=attn_dropout,
#                     norm_cfg=norm_cfg
#                 ) for _ in range(num_attn_layers)]
#             )

#             # 特征重组投影（恢复原通道）
#             stage_module['recon_proj'] = ConvModule(
#                 in_channels=attn_embed_dims,
#                 out_channels=patch_dim,
#                 kernel_size=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=conv_norm_cfg,
#                 act_cfg=act_cfg
#             )

#             self.stage_modules.append(stage_module)

#     def _split_into_patches(self, feat, patch_size):  #　将形状为 (B, C, H, W) 的特征图　转换为形状为 (B, N, P) 的 Patch 序列
#         B, C, H, W = feat.shape
#         p_h, p_w = patch_size

#         # Padding补全（确保能被patch尺寸整除）
#         pad_h = (p_h - H % p_h) % p_h
#         pad_w = (p_w - W % p_w) % p_w
#         if pad_h > 0 or pad_w > 0:
#             feat = F.pad(feat, (0, pad_w, 0, pad_h), mode='reflect') # padding 补全特征图  可能有的特征图 被patch分割后未除尽 

#         # 切分patch并展平为序列
#         H_pad, W_pad = feat.shape[2], feat.shape[3]
#         num_patches_h = H_pad // p_h
#         num_patches_w = W_pad // p_w
#         num_patches = num_patches_h * num_patches_w
        
#         patches = feat.unfold(2, p_h, p_h).unfold(3, p_w, p_w)  # (B, C, num_h, num_w, p_h, p_w)
#         patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, num_h, num_w, C, p_h, p_w)
#         patch_seq = patches.reshape(B, num_patches, C * p_h * p_w)  # (B, num_patches, patch_dim)

#         return patch_seq, pad_h, pad_w, num_patches_h, num_patches_w, num_patches

#     def _reconstruct_from_patches(self, patch_seq, patch_size, pad_h, pad_w, num_patches_h, num_patches_w, orig_H, orig_W):
#         B, _, patch_dim = patch_seq.shape
#         p_h, p_w = patch_size
#         C = patch_dim // (p_h * p_w)

#         # Patch序列→网格
#         patches = patch_seq.reshape(B, num_patches_h, num_patches_w, C, p_h, p_w)
#         patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, num_h, p_h, num_w, p_w)

#         # 拼接为完整特征
#         H_pad = num_patches_h * p_h
#         W_pad = num_patches_w * p_w
#         recon_feat = patches.reshape(B, C, H_pad, W_pad)

#         # 移除Padding
#         if pad_h > 0:
#             recon_feat = recon_feat[:, :, :-pad_h, :]
#         if pad_w > 0:
#             recon_feat = recon_feat[:, :, :, :-pad_w]

#         # 确保与原始尺寸一致
#         recon_feat = F.interpolate(recon_feat, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
#         return recon_feat
#         # 重建特征图 
#     def forward(self, img_feats, view_img_feats):
#         assert len(img_feats) == len(view_img_feats) == len(self.stage_modules), \
#             "特征阶段数、view特征阶段数、模块数必须一致" # 保证尺度的一致性 
        
#         # 直接使用原始view特征进行交叉注意力融合
#         enhanced_img_feats = []
#         for img_feat, view_feat, stage_mod, p_size in zip(
#             img_feats, view_img_feats, self.stage_modules, self.patch_sizes
#         ):
#             B, C, orig_H, orig_W = img_feat.shape

#             # 1. 切分Patch（主模态用原始img_feat，辅助模态用原始view_feat）
#             img_patch_seq, img_pad_h, img_pad_w, num_h, num_w, num_patches = self._split_into_patches(img_feat, p_size)
#             view_patch_seq, _, _, _, _, _ = self._split_into_patches(view_feat, p_size)
#             patch_dim = img_patch_seq.shape[2]

#             # 2. Patch维度适配
#             B_num_patches = B * num_patches
#             img_patch_proj = stage_mod['patch_proj'](img_patch_seq.reshape(B_num_patches, patch_dim, 1, 1))
#             img_patch_proj = img_patch_proj.squeeze(-1).squeeze(-1).reshape(B, -1, self.attn_embed_dims)
#             view_patch_proj = stage_mod['patch_proj'](view_patch_seq.reshape(B_num_patches, patch_dim, 1, 1))
#             view_patch_proj = view_patch_proj.squeeze(-1).squeeze(-1).reshape(B, -1, self.attn_embed_dims)

#             # 3. 交叉注意力增强（核心逻辑保留）
#             img_patch_enhanced = stage_mod['attn_blocks'](img_patch_proj, view_patch_proj)

#             # 4. 重组与残差输出
#             img_patch_recon = stage_mod['recon_proj'](img_patch_enhanced.reshape(B_num_patches, self.attn_embed_dims, 1, 1))
#             img_patch_recon = img_patch_recon.squeeze(-1).squeeze(-1).reshape(B, -1, patch_dim)
            
#             img_recon = self._reconstruct_from_patches(
#                 patch_seq=img_patch_recon,
#                 patch_size=p_size,
#                 pad_h=img_pad_h,
#                 pad_w=img_pad_w,
#                 num_patches_h=num_h,
#                 num_patches_w=num_w,
#                 orig_H=orig_H,
#                 orig_W=orig_W
#             )

#             enhanced_feat = img_recon + img_feat  # 残差连接保留
#             enhanced_img_feats.append(enhanced_feat)

#         return enhanced_img_feats


def build_fuse_neck(cfg, default_args=None):
    return build_from_cfg(cfg, FUSE_NECKS, default_args)


# 测试功能（输入输出形状不变）
if __name__ == "__main__":
    # 模拟输入：3阶段特征
    img_feats = [
        torch.randn(1, 512, 250, 250),
        torch.randn(1, 1024, 125, 125),
        torch.randn(1, 2048, 63, 63)
    ]
    view_feats = [f.clone() for f in img_feats]  # 模拟view模态特征

    # 初始化融合模块（已移除DCN，保留交叉注意力）
    fusion = CrossAttnFeatEnhancer(in_channels=[512, 1024, 2048])

    # 前向传播
    fused_feats = fusion(img_feats, view_feats)

    # 验证输出形状与输入一致
    print("输入img_feats形状:", [f.shape for f in img_feats])
    print("输出fused_feats形状:", [f.shape for f in fused_feats])