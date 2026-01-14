import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2  # 用于热力图平滑和缩放
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.transformer import DeformableDetrTransformer, inverse_sigmoid
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from .decoder import VMADetectionTransformerDecoder

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class SplitModalityDecoder(VMADetectionTransformerDecoder):
    def __init__(self, *args, split_layer_index=3, use_gating=True, use_gradual=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_layer_index = split_layer_index
        self.use_gating = use_gating
        self.use_gradual = use_gradual
        # self.test = False

        # [DEBUG ADDITION]
        self.debug_step = 0
        self.debug_dir = "debug_vis_1/attention_check"
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
        except Exception:
            pass
        
        # Gating module: Learn to filter view features based on lidar features
        if self.use_gating:
            embed_dims = kwargs.get('embed_dims', 256)
            self.gating_module = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
                nn.Sigmoid()
            )

            # Initialize weights
            for m in self.gating_module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # [DEBUG ADDITION]
    def visualize_reference_points(self, reference_points, layer_idx, img_metas=None, feature_map=None, feature_map_view=None, spatial_shapes=None):
        """
        reference_points: (BS, Num_Query, 2)
        feature_map: (BS, Len_Seq, C)  Lidar
        feature_map_view: (BS, Len_Seq, C) View
        spatial_shapes: (Num_Levels, 2)
        """
        # Save every 100 steps
        if self.debug_step % 1000 != 0:
            return

        try:
            # --- 1. Reference Points Heatmap ---
            pts = reference_points[0].detach().cpu().numpy()
            
            # 使用高斯模糊生成热力图
            img_h, img_w = 200, 200
            heatmap = np.zeros((img_h, img_w), dtype=np.float32)
            
            px = (pts[:, 0] * img_w).astype(np.int32)
            py = (pts[:, 1] * img_h).astype(np.int32)
            
            valid_mask = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
            px = px[valid_mask]
            py = py[valid_mask]
            
            for x, y in zip(px, py):
                heatmap[y, x] += 1
            
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            heatmap = heatmap / (heatmap.max() + 1e-9)
            
            # helper to process feature map
            def process_feat(feat_map, s_shapes):
                feat_img = None
                status = ""
                if feat_map is not None and s_shapes is not None:
                    if feat_map.shape[1] != feat_map.shape[0] and feat_map.shape[0] > feat_map.shape[1]: 
                         seq_dim = 0
                    else:
                         seq_dim = 1
                    
                    current_len = feat_map.shape[seq_dim]
                    H, W = s_shapes[0].tolist()
                    feat_len = H * W
                    
                    if current_len >= feat_len:
                        if seq_dim == 0:
                            feat = feat_map[:feat_len, 0, :].detach().cpu()
                        else:
                            feat = feat_map[0, :feat_len, :].detach().cpu()
                        
                        feat_norm = torch.norm(feat, dim=1).numpy()
                        try:
                            feat_img = feat_norm.reshape(H, W)
                            feat_img = cv2.resize(feat_img, (img_w, img_h))
                            # Normalize for better vis
                            feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min() + 1e-9)
                        except ValueError:
                             status = "Reshape Fail"
                    else:
                        status = "Size Mismatch"
                return feat_img, status

            feat_img_lidar, status_lidar = process_feat(feature_map, spatial_shapes)
            feat_img_view, status_view = process_feat(feature_map_view, spatial_shapes)

            # --- Plotting ---
            plt.figure(figsize=(18, 6))
            
            # Subplot 1: Ref Points Heatmap
            plt.subplot(1, 3, 1)
            plt.imshow(heatmap, cmap='jet', extent=[0, 1, 1, 0])
            plt.title(f"Ref Points Density (L{layer_idx})")
            plt.colorbar(fraction=0.046, pad=0.04)
            
            # Subplot 2: Lidar Feature Map
            plt.subplot(1, 3, 2)
            if feat_img_lidar is not None:
                plt.imshow(feat_img_lidar, cmap='viridis', extent=[0, 1, 1, 0])
                plt.imshow(heatmap, cmap='jet', alpha=0.3, extent=[0, 1, 1, 0]) 
                plt.title("Lidar (Lvl 0) + Refs")
            else:
                plt.text(0.5, 0.5, status_lidar or "No Lidar", ha='center')
                plt.title("Lidar Skipped")

            # Subplot 3: View Feature Map
            plt.subplot(1, 3, 3)
            if feat_img_view is not None:
                plt.imshow(feat_img_view, cmap='viridis', extent=[0, 1, 1, 0])
                plt.imshow(heatmap, cmap='jet', alpha=0.3, extent=[0, 1, 1, 0]) 
                plt.title("View (Lvl 0) + Refs")
            else:
                plt.text(0.5, 0.5, status_view or "No View", ha='center')
                plt.title("View Skipped")
                
            plt.suptitle(f"Step {self.debug_step} Layer {layer_idx}")
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_heatmap.png")
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Vis error: {e}")

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                value_view=None,
                **kwargs):
        
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        # Capture the primary value (usually lidar) from kwargs
        value_lidar = kwargs.get('value', None)
        
        for lid, layer in enumerate(self.layers):
            current_value = value_lidar

            refined_view = None # 初始化 refined_view
            if value_view is not None:
                # 1. Gating / Noise Filtering
                refined_view = value_view
                if self.use_gating and value_lidar is not None:
                     # ... existing gating logig ...
                    # value_lidar: (BS, Num_Keys, C)
                    # value_view: (BS, Num_Keys, C)
                    # Concat along channel dimension
                    cat_feat = torch.cat([value_lidar, value_view], dim=-1)
                    gate = self.gating_module(cat_feat)
                    refined_view = value_view * gate

            # [DEBUG ADDITION]
            if self.training:
                 # Pass both Lidar (value_lidar) and View (refined_view OR value_view)
                 # We prefer showing refined_view if gating is ON, to see what actually goes into fusion
                 vis_view = refined_view if refined_view is not None else value_view
                 self.visualize_reference_points(
                     reference_points, 
                     lid, 
                     feature_map=kwargs.get('value', None), # This is usually 'current_value' from PREVIOUS layer logic? Wait, kwargs['value'] is updated at end of loop.
                     # Actually 'value_lidar' is constant across layers as "Memory".
                     # The decoder usually takes 'memory' as input. 
                     # Let's visualize the RAW Inputs to this layer fusion.
                     # Lidar = value_lidar
                     feature_map_view=vis_view,
                     spatial_shapes=kwargs.get('spatial_shapes', None)
                )

            if value_view is not None:
                # ... Fusion Logic (Gradual/Hard Split) ...
                # 2. Gradual Fusion or Hard Split
                if self.use_gradual:
                    # Calculate weight: 0 at first layer, 1 at last layer
                    num_layers = len(self.layers)
                    # progress = lid / (num_layers - 1) if num_layers > 1 else 1.0 # Linear 0->1
                    
                    # Implementation: Linear transition scheme (Lidar Dominant)
                    # View weight increases: 0.0 -> 1.0
                    view_weight = float(lid) / float(num_layers - 1)
                    view_weight = min(max(view_weight, 0.0), 1.0)
                    
                    # CHANGED: Lidar weight stays 1.0 instead of decreasing.
                    # This ensures Lidar (Geometric Base) is never lost.
                    lidar_weight = 1.0
                    
                    # Fused value: Lidar + Scaled View (Residual connection style)
                    if value_lidar is not None:
                        current_value = lidar_weight * value_lidar + view_weight * refined_view
                    else:
                        current_value = refined_view
                        
                else:
                    # Legacy Hard Split Logic
                    if lid >= self.split_layer_index:
                        current_value = refined_view
            
            # Update kwargs locally for this layer call
            kwargs['value'] = current_value

            reference_points_input = reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
            
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # [DEBUG ADDITION]
        if self.training:
            self.debug_step += 1

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class SplitModalityTransformer(DeformableDetrTransformer):
    def __init__(self, *args, encoder_view=None, **kwargs):
        super(SplitModalityTransformer, self).__init__(*args, **kwargs)
        if encoder_view is not None:
            self.encoder_view = build_transformer_layer_sequence(encoder_view)
        else:
            self.encoder_view = None

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                mlvl_feats_view=None,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        
        assert self.as_two_stage is False, "SplitModalityTransformer only supports non-two-stage mode for now."

        bs = mlvl_feats[0].size(0)
        
        # --- 1. Prepare Inputs for Standard Flow (Lidar) ---
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        # DEBUG PRINTS
        # print(f"DEBUG: spatial_shapes={spatial_shapes}")
        # print(f"DEBUG: feat_flatten.shape={feat_flatten.shape}")
        # total_pixels = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()
        # print(f"DEBUG: calculated total_pixels={total_pixels}")
        # assert total_pixels == feat_flatten.shape[1], f"Mismatch! {total_pixels} vs {feat_flatten.shape[1]}"

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # Calculate encoder reference points
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat_flatten.device)

        # --- 2. Encoder ---
        memory = feat_flatten
        if self.encoder is not None:
             memory = self.encoder(
                 query=feat_flatten.permute(1, 0, 2),
                 key=None,
                 value=None,
                 query_pos=lvl_pos_embed_flatten.permute(1, 0, 2),
                 query_key_padding_mask=mask_flatten,
                 spatial_shapes=spatial_shapes,
                 level_start_index=level_start_index,
                 valid_ratios=valid_ratios,
                 reference_points=reference_points,
                 **kwargs
             )
        
        # --- 3. Process View Feats ---
        memory_view = None
        if mlvl_feats_view is not None:
            feat_flatten_view = []
            for lvl, feat in enumerate(mlvl_feats_view):
                # Assume alignment with Lidar
                feat = feat.flatten(2).transpose(1, 2)
                # Note: We reuse the same level embeddings logic, but applied to view features
                # lvl_pos_embed is already calculated above and matches if shapes match
                
                # However, for the encoder input, we generally want the feature + level_embed (sometimes)
                # But DeformableDetrTransformerEncoder usually takes 'query' (feature) and 'query_pos' (pos + lvl)
                # In the loop above `lvl_pos_embed` = `pos_embed` + `level_embed`.
                # We can reuse `lvl_pos_embed_flatten` because spatial shapes are the same.
                
                feat_flatten_view.append(feat)
            
            # Combine levels
            feat_flatten_view = torch.cat(feat_flatten_view, 1) # (BS, Total_Len, C)
            
            # Pass through View Encoder if it exists
            if self.encoder_view is not None:
                # Reuse geometry info from Lidar branch since we enforced alignment
                memory_view = self.encoder_view(
                    query=feat_flatten_view.permute(1, 0, 2), # (Len, BS, C)
                    key=None,
                    value=None,
                    query_pos=lvl_pos_embed_flatten.permute(1, 0, 2), # Reuse pos embeds
                    query_key_padding_mask=mask_flatten,              # Reuse masks
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points,
                    **kwargs
                )
            else:
                memory_view = feat_flatten_view.permute(1, 0, 2)

        # --- 4. Decoder ---
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory, 
            value_view=memory_view, 
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)
            
        return inter_states, init_reference_out, inter_references, None, None
