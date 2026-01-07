from mmcv.runner.base_module import BaseModule
# from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.transformer import DeformableDetrTransformer  # 导入底层Transformer
from mmcv.utils import Registry, build_from_cfg
# from mmdet3d.registry import TRANSFORMERS  # 优先使用 mmdet3d 的注册表
from mmdet.models.utils.builder import TRANSFORMER

import sys
from pathlib import Path
# sys.path.insert(0, '/homes/zhangzijian/vma-dev')  # 强制置顶

import torch
sys.path.append(str(Path(__file__).resolve().parents[4]))
# from projects.mmdet3d_plugin.vma.builder import FUSE_NECKS
# from projects.mmdet3d_plugin.vma.builder import TRANSFORMERS   # 导入自定义注册表

@TRANSFORMER.register_module()  # 注册为Transformer模块，配置中可直接调用
class VMATransformer(BaseModule):
    def __init__(self,
                 encoder=None,          # 原有：lidar_map的编码器
                 encoder_view=None,     # 新增：view_map的编码器
                 decoder=None,          # 原有：VMADetectionTransformerDecoder
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 init_cfg=None,** kwargs):
        super().__init__(init_cfg=init_cfg)
        # 1. 复用底层DeformableDetrTransformer的核心逻辑
        self.base_transformer = DeformableDetrTransformer(
            encoder=encoder,
            decoder=decoder,
            as_two_stage=as_two_stage,
            num_feature_levels=num_feature_levels,
            two_stage_num_proposals=two_stage_num_proposals,** kwargs
        )
        # 2. 新增：初始化view_map的编码器（复用底层DetrTransformerEncoder）
        self.encoder_view = build_transformer_layer_sequence(encoder_view) if encoder_view else None
        self.embed_dims = self.base_transformer.embed_dims  # 与底层保持维度一致

    def init_weights(self):
        # 复用底层初始化逻辑
        self.base_transformer.init_weights()
        # 额外初始化view编码器（若有）
        if self.encoder_view is not None:
            for p in self.encoder_view.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

    def forward(self,
                mlvl_feats,          # 原有：lidar_map的多尺度特征
                mlvl_feats_view,     # 新增：view_map的多尺度特征
                mlvl_masks,          # 原有：lidar_map的掩码
                mlvl_masks_view,     # 新增：view_map的掩码
                query_embed,         # 原有：查询嵌入
                mlvl_pos_embeds,     # 原有：lidar_map的位置编码
                mlvl_pos_embeds_view,# 新增：view_map的位置编码
                reg_branches=None,
                cls_branches=None,** kwargs):
        """
        新增输入参数：
        - mlvl_feats_view: view_map的多尺度特征，list[Tensor]，每个元素 (bs, c, h, w)
        - mlvl_masks_view: view_map的掩码，list[Tensor]，每个元素 (bs, h, w)
        - mlvl_pos_embeds_view: view_map的位置编码，list[Tensor]，每个元素 (bs, c, h, w)
        """
        # -------------------------- 1. 新增：计算view_map的编码结果（memory_view） --------------------------
        memory_view = None
        if self.encoder_view is not None and mlvl_feats_view is not None:
            # 复用底层DeformableDetrTransformer的“多尺度特征转序列”逻辑
            feat_flatten_view = []
            mask_flatten_view = []
            pos_flatten_view = []
            spatial_shapes_view = []
            for lvl, (feat, mask, pos) in enumerate(zip(mlvl_feats_view, mlvl_masks_view, mlvl_pos_embeds_view)):
                bs, c, h, w = feat.shape
                spatial_shapes_view.append((h, w))
                # 特征扁平化：(bs, c, h, w) → (bs, h*w, c)
                feat_flat = feat.flatten(2).transpose(1, 2)
                # 掩码扁平化：(bs, h, w) → (bs, h*w)
                mask_flat = mask.flatten(1)
                # 位置编码扁平化：(bs, c, h, w) → (bs, h*w, c)
                pos_flat = pos.flatten(2).transpose(1, 2)
                # 叠加层级嵌入（若有）
                if hasattr(self.base_transformer, 'level_embeds'):
                    pos_flat += self.base_transformer.level_embeds[lvl].view(1, 1, -1).to(pos_flat.dtype)
                # 收集结果
                feat_flatten_view.append(feat_flat)
                mask_flatten_view.append(mask_flat)
                pos_flatten_view.append(pos_flat)
            # 拼接多尺度特征
            feat_flatten_view = torch.cat(feat_flatten_view, 1)
            mask_flatten_view = torch.cat(mask_flatten_view, 1)
            pos_flatten_view = torch.cat(pos_flatten_view, 1)
            spatial_shapes_view = torch.tensor(spatial_shapes_view, dtype=torch.long, device=feat_flatten_view.device)
            level_start_index_view = torch.cat((spatial_shapes_view.new_zeros((1,)), spatial_shapes_view.prod(1).cumsum(0)[:-1]))

            # view编码器前向：得到memory_view（view_map的全局特征序列）
            memory_view = self.encoder_view(
                query=feat_flatten_view.permute(1, 0, 2),  # 转成 (h*w, bs, c)，适配编码器输入
                query_pos=pos_flatten_view.permute(1, 0, 2),
                query_key_padding_mask=mask_flatten_view,
                spatial_shapes=spatial_shapes_view,
                level_start_index=level_start_index_view,** kwargs
            ).permute(1, 0, 2)  # 转回 (bs, h*w, c)，后续传递给解码器

        # -------------------------- 2. 复用底层逻辑：计算lidar_map的编码结果 --------------------------
        # 调用底层DeformableDetrTransformer的forward，得到lidar相关结果
        lidar_outputs = self.base_transformer(
            mlvl_feats=mlvl_feats,
            mlvl_masks=mlvl_masks,
            query_embed=query_embed,
            mlvl_pos_embeds=mlvl_pos_embeds,
            reg_branches=reg_branches,
            cls_branches=cls_branches,** kwargs
        )
        inter_states, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact = lidar_outputs

        # -------------------------- 3. 新增：将memory_view传递给解码器（通过args打包） --------------------------
        # 若解码器是你修改的VMADetectionTransformerDecoder，需将memory_view传入其forward
        # 这里直接在返回结果中增加memory_view，后续在检测头中调用解码器时使用
        return inter_states, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, memory_view

# def build_vma_transformer(cfg, default_args=None):
#     return build_from_cfg(cfg, TRANSFORMERS , default_args)

if __name__ == '__main__':
    from projects.mmdet3d_plugin.vma.builder import VMA_TRANSFORMER
    # 检查是否注册成功
    print("VMATransformer 是否在 VMA_TRANSFORMER 注册表中：", 
          'VMATransformer' in VMA_TRANSFORMER.module_dict)  # 应返回 True