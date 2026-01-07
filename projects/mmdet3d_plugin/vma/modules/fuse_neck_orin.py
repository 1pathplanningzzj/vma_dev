import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import Registry, build_from_cfg
from torchvision.ops import DeformConv2d
from mmcv.cnn import ConvModule, build_norm_layer
#from ..builder import FUSE_NECKS
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径（需根据你的目录结构调整）
# 假设 fuse_neck.py 路径为：VMA-main/projects/mmdet3d_plugin/vma/modules/fuse_neck.py
# 则项目根目录为 VMA-main/
sys.path.append(str(Path(__file__).resolve().parents[4]))  # 向上4级目录找到根目录

# 将相对导入改为绝对导入
from projects.mmdet3d_plugin.vma.builder import FUSE_NECKS

#@FUSE_NECKS.register_module()
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

class DCN(nn.Module):
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
        '''self.fusion_convs = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, 1) for ch in in_channels
        ])'''
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
            
            
            fused_feats.append(deformed_view)
            
        return fused_feats

class CrossAttnBlock(nn.Module):
    """单一层Cross-Attention单元，用于叠加"""
    def __init__(self, attn_embed_dims, num_heads, attn_dropout, norm_cfg):
        super().__init__()
        self.attn_embed_dims = attn_embed_dims
        # Query/Key/Value投影（确保维度一致）
        self.q_proj = nn.Linear(attn_embed_dims, attn_embed_dims)
        self.k_proj = nn.Linear(attn_embed_dims, attn_embed_dims)
        self.v_proj = nn.Linear(attn_embed_dims, attn_embed_dims)
        
        # Cross-Attention层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_embed_dims,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
        # 归一化与残差适配
        self.norm1 = build_norm_layer(norm_cfg, attn_embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, attn_embed_dims)[1]
        self.ffn = nn.Sequential(
            nn.Linear(attn_embed_dims, attn_embed_dims * 2),
            nn.ReLU(),
            nn.Linear(attn_embed_dims * 2, attn_embed_dims)
        )

    def forward(self, img_patch_seq, view_patch_seq):
        """
        Args:
            img_patch_seq: img的patch序列，形状(B, num_patches, patch_dim)
            view_patch_seq: view_img的patch序列，形状同上
        Returns:
            经过一层Cross-Attention的img_patch_seq
        """
        # 1. 投影与Attention
        q = self.q_proj(img_patch_seq)  # (B, N, D)
        k = self.k_proj(view_patch_seq)
        v = self.v_proj(view_patch_seq)
        
        attn_out, _ = self.cross_attn(query=q, key=k, value=v, need_weights=False)
        # 残差1：Attention输出 + 原img序列
        img_patch_seq = self.norm1(attn_out + img_patch_seq)
        
        # 2.  Feed-Forward Network（增强非线性表达）
        ffn_out = self.ffn(img_patch_seq)
        # 残差2：FFN输出 + Attention后的序列
        img_patch_seq = self.norm2(ffn_out + img_patch_seq)
        
        return img_patch_seq

class SequentialMultiInput(nn.Module):
    """支持多输入参数的Sequential容器"""
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, y):
        """将x和y依次传递给每个模块"""
        for module in self.modules_list:
            x = module(x, y)
        return x

@FUSE_NECKS.register_module()
class CrossAttnFeatEnhancer(nn.Module):
    """
    Patch级三层Cross-Attention融合器：
    1. 按阶段切分patch（尺寸[4,2,1]）；2. 三层Cross-Attention叠加；3. 重组特征并残差增强
    """
    def __init__(self,
                 in_channels,  # 各阶段输入通道，如[512, 1024, 2048]
                 patch_sizes=[[16,16], [8,8], [4,4]],  # 各阶段patch尺寸（H×W），对应[4,2,1]
                 num_attn_layers=3,  # 叠加三层Cross-Attention
                 attn_embed_dims=256,  # Attention隐藏层维度
                 num_heads=8,  # 多头注意力头数
                 attn_dropout=0.1,
                 norm_cfg=dict(type='LN'),
                 conv_norm_cfg=dict(type='BN'),
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.patch_sizes = patch_sizes
        self.num_attn_layers = num_attn_layers
        self.attn_embed_dims = attn_embed_dims
        self.num_heads = num_heads
        self.defor = DCN(in_channels=in_channels)
        assert len(in_channels) == len(patch_sizes), "通道数与patch尺寸数必须一致"
        assert attn_embed_dims % num_heads == 0, "attn_embed_dims需被num_heads整除"

        # 构建每个阶段的处理模块（patch切分→Attention→重组）
        self.stage_modules = nn.ModuleList()
        for idx, (in_dim, p_size) in enumerate(zip(in_channels, patch_sizes)):
            p_h, p_w = p_size
            stage_module = nn.ModuleDict()

            # 1. Patch维度投影：将patch_dim（C×p_h×p_w）适配到attn_embed_dims
            patch_dim = in_dim * p_h * p_w  # 单个patch的总维度
            stage_module['patch_proj'] = ConvModule(
                in_channels=patch_dim,
                out_channels=attn_embed_dims,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None
            )

            # 2. 三层Cross-Attention叠加
            stage_module['attn_blocks'] = SequentialMultiInput(
                *[CrossAttnBlock(
                    attn_embed_dims=attn_embed_dims,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    norm_cfg=norm_cfg
                ) for _ in range(num_attn_layers)]
            )

            # 3. 特征重组投影：将处理后的patch序列映射回原通道
            stage_module['recon_proj'] = ConvModule(
                in_channels=attn_embed_dims,
                out_channels=patch_dim,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=conv_norm_cfg,
                act_cfg=act_cfg
            )

            self.stage_modules.append(stage_module)

    def _split_into_patches(self, feat, patch_size):
            B, C, H, W = feat.shape
            p_h, p_w = patch_size

            # Padding补全（确保能被patch尺寸整除）
            pad_h = (p_h - H % p_h) % p_h
            pad_w = (p_w - W % p_w) % p_w
            if pad_h > 0 or pad_w > 0:
                feat = F.pad(feat, (0, pad_w, 0, pad_h), mode='reflect')

            # 切分patch并展平为序列
            H_pad, W_pad = feat.shape[2], feat.shape[3]
            num_patches_h = H_pad // p_h
            num_patches_w = W_pad // p_w
            num_patches = num_patches_h * num_patches_w  # 总patch数
            
            patches = feat.unfold(2, p_h, p_h).unfold(3, p_w, p_w)  # (B, C, num_h, num_w, p_h, p_w)
            patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, num_h, num_w, C, p_h, p_w)
            patch_seq = patches.reshape(B, num_patches, C * p_h * p_w)  # (B, num_patches, patch_dim)

            return patch_seq, pad_h, pad_w, num_patches_h, num_patches_w, num_patches  # 新增返回num_patches

    def _reconstruct_from_patches(self, patch_seq, patch_size, pad_h, pad_w, num_patches_h, num_patches_w, orig_H, orig_W):
        """
        将处理后的patch序列重组回原特征形状，移除padding
        Args:
            patch_seq: 处理后的patch序列，(B, num_patches, C×p_h×p_w)
            patch_size: (p_h, p_w)
            pad_h/pad_w: 之前补全的像素数
            num_patches_h/num_patches_w: patch行数/列数
            orig_H/orig_W: 原始特征的H/W（未padding前）
        Returns:
            recon_feat: 重组后的特征，(B, C, orig_H, orig_W)
        """
        B, _, patch_dim = patch_seq.shape
        p_h, p_w = patch_size
        C = patch_dim // (p_h * p_w)  # 恢复原通道数

        # Step 1: patch序列→patch网格（B, num_h, num_w, C, p_h, p_w）
        patches = patch_seq.reshape(B, num_patches_h, num_patches_w, C, p_h, p_w)
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, num_h, p_h, num_w, p_w)

        # Step 2: 拼接patch→完整特征（B, C, H_pad, W_pad）
        H_pad = num_patches_h * p_h
        W_pad = num_patches_w * p_w
        recon_feat = patches.reshape(B, C, H_pad, W_pad)

        # Step 3: 移除padding，恢复原尺寸
        if pad_h > 0:
            recon_feat = recon_feat[:, :, :-pad_h, :]
        if pad_w > 0:
            recon_feat = recon_feat[:, :, :, :-pad_w]

        # 确保重组后尺寸与原始一致（防止极端情况）
        recon_feat = F.interpolate(recon_feat, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        return recon_feat

    def forward(self, img_feats, view_img_feats):
        """
        Args:
            img_feats: list[Tensor]，img_backbone输出，每个元素(B, C, H, W)
            view_img_feats: list[Tensor]，view_img_backbone输出，与img_feats对应
        Returns:
            enhanced_img_feats: list[Tensor]，增强后的img_feats，尺寸与输入一致
        """
        assert len(img_feats) == len(view_img_feats) == len(self.stage_modules), \
            "特征阶段数、view特征阶段数、模块数必须一致"
        view_img_feats = self.defor(img_feats,view_img_feats)
        enhanced_img_feats = []
        for img_feat, view_feat, stage_mod, p_size in zip(
            img_feats, view_img_feats, self.stage_modules, self.patch_sizes
        ):
            B, C, orig_H, orig_W = img_feat.shape  # 原始特征尺寸（未padding）

            # -------------------------- 1. 切分img和view的patch --------------------------
            # img_feat→patch序列
            img_patch_seq, img_pad_h, img_pad_w, num_h, num_w, num_patches = self._split_into_patches(img_feat, p_size)
            view_patch_seq, _, _, _, _, _ = self._split_into_patches(view_feat, p_size)
            patch_dim = img_patch_seq.shape[2]  # 获取单个patch的维度

                # 2. Patch维度适配（核心修改：正确计算B_num_patches）
            B_num_patches = B * num_patches  # 总patch数 = 批次×每个样本的patch数
            img_patch_proj = stage_mod['patch_proj'](img_patch_seq.reshape(B_num_patches, patch_dim, 1, 1))
            img_patch_proj = img_patch_proj.squeeze(-1).squeeze(-1).reshape(B, -1, self.attn_embed_dims)
            
            view_patch_proj = stage_mod['patch_proj'](view_patch_seq.reshape(B_num_patches, patch_dim, 1, 1))
            view_patch_proj = view_patch_proj.squeeze(-1).squeeze(-1).reshape(B, -1, self.attn_embed_dims)

            # -------------------------- 3. 三层Cross-Attention叠加 --------------------------
            img_patch_enhanced = stage_mod['attn_blocks'](img_patch_proj, view_patch_proj)

            # -------------------------- 4. 重组patch→原特征形状 --------------------------
            # 投影回patch_dim：(B, num_patches, attn_embed_dims) → (B, num_patches, patch_dim)
            img_patch_recon = stage_mod['recon_proj'](img_patch_enhanced.reshape(B_num_patches, self.attn_embed_dims, 1, 1))
            img_patch_recon = img_patch_recon.squeeze(-1).squeeze(-1).reshape(B, -1, patch_dim)
            
            # 重组为特征图
            img_recon = self._reconstruct_from_patches(
                patch_seq=img_patch_recon,
                patch_size=p_size,
                pad_h=img_pad_h,
                pad_w=img_pad_w,
                num_patches_h=num_h,
                num_patches_w=num_w,
                orig_H=orig_H,
                orig_W=orig_W
            )

            # -------------------------- 5. 残差增强 --------------------------
            enhanced_feat = img_recon + img_feat  # 保留原img特征信息
            enhanced_img_feats.append(enhanced_feat)

        return enhanced_img_feats


def build_fuse_neck(cfg, default_args=None):
    """根据配置构建融合编码器"""
    return build_from_cfg(cfg, FUSE_NECKS, default_args)
    
if __name__ == "__main__":
    img_feats = [
        torch.randn(2, 512, 250, 250),   # stage1
        torch.randn(2, 1024, 125, 125),   # stage2 
        torch.randn(2, 2048, 63, 63)     # stage3
    ]
    view_feats = [f.clone() for f in img_feats]  # 模拟view特征

    # 初始化融合模块
    fusion = CrossAttnFeatEnhancer(in_channels=[512, 1024, 2048])

    # 前向传播
    fused_feats = fusion(img_feats, view_feats)

    # 输出shape与输入保持一致
    print([f.shape for f in fused_feats]) 