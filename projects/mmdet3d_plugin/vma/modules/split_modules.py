# date 2026-01-15
# author: zijian
# email: zhangzijian@trunk.tech
# description: This file is the implementation of the SplitModalityDecoder class.
# It is used to decode the features of the lidar and view modalities.
# It is also used to fuse the features of the lidar and view modalities.
# It is also used to predict the bounding boxes of the objects.
# It is also used to predict the classes of the objects.
# It is also used to predict the scores of the objects.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2  # 用于热力图平滑和缩放
try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None  # 如果scipy不可用，可视化时会跳过相关功能
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.transformer import DeformableDetrTransformer, inverse_sigmoid
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from .decoder import VMADetectionTransformerDecoder

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class SplitModalityDecoder(VMADetectionTransformerDecoder):
    def __init__(self, *args, split_layer_index=3, use_gating=True, use_gradual=True, 
                 use_dual_cross_attn=False, 
                 use_adaptive_weights=False,
                 use_modality_interaction=False,
                 use_conditional_fusion=False,
                 lidar_first=True,  # [NEW] Lidar first (main modality), then View (auxiliary)
                 gate_temperature=1.0,  # [FIX] Add as explicit parameter
                 embed_dims=256,  # [FIX] Add as explicit parameter
                 use_adaptive_gating=True,  # [NEW] 是否启用基于空间方差的动态Gating
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.split_layer_index = split_layer_index
        self.use_gating = use_gating
        self.use_gradual = use_gradual
        # [NEW] Enable dual cross attention: query attends to both View and Lidar separately
        self.use_dual_cross_attn = use_dual_cross_attn
        self.use_adaptive_weights = use_adaptive_weights
        self.use_modality_interaction = use_modality_interaction
        self.use_conditional_fusion = use_conditional_fusion
        self.lidar_first = lidar_first  # Lidar is main modality, process first
        self.gate_temperature = gate_temperature  # [FIX] Store gate_temperature
        self.use_adaptive_gating = use_adaptive_gating  # [NEW] 是否启用基于空间方差的动态Gating
        
        # [FIX] Get embed_dims from kwargs if provided, otherwise use parameter value
        if 'embed_dims' in kwargs:
            embed_dims = kwargs.pop('embed_dims')  # Use kwargs value and remove it
        # Otherwise use the parameter value (already set as embed_dims=256)
        
        # [NEW] Adaptive weight learning module (per-query spatial adaptive)
        if self.use_adaptive_weights:
            # [FIX] Changed to per-query prediction instead of global
            self.adaptive_weight_module = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.LayerNorm(embed_dims),  # [FIX] Add LayerNorm for stability
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims // 2),
                nn.LayerNorm(embed_dims // 2),  # [FIX] Add LayerNorm for stability
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims // 2, 2),  # Output: [lidar_weight, view_weight] per query
                nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
            )
            # Initialize to favor Lidar (main modality)
            linear_layers = [m for m in self.adaptive_weight_module.modules() if isinstance(m, nn.Linear)]
            for idx, m in enumerate(linear_layers):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if idx == len(linear_layers) - 1:  # Last Linear layer (before Softmax)
                        # Bias to favor Lidar: [lidar_bias, view_bias] = [1.0, -1.0]
                        with torch.no_grad():
                            m.bias.data = torch.tensor([1.0, -1.0], dtype=m.bias.dtype, device=m.bias.device)
                    else:
                        nn.init.constant_(m.bias, 0)
        
        # [NEW] Modality interaction module (for cross-modal guidance)
        if self.use_modality_interaction:
            self.modality_interaction = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims)
            )
            for m in self.modality_interaction.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # [NEW] Conditional fusion module (query-dependent modality selection)
        if self.use_conditional_fusion:
            self.conditional_fusion = nn.Sequential(
                nn.Linear(embed_dims, embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims // 2, 2),  # Output: [lidar_gate, view_gate]
                nn.Sigmoid()
            )
            for m in self.conditional_fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        # self.test = False

        # [DEBUG ADDITION]
        self.debug_step = 0
        self.debug_dir = "debug_vis_gt_mask_5/attention_check"
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
        except Exception:
            pass
        
        # Gating module: Learn to filter view features based on lidar features
        if self.use_gating:
            # embed_dims is already set above from parameter or kwargs
            self.gating_module = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
                nn.Sigmoid()
            )
            
            # [IMPROVED] Gate temperature is already set as self.gate_temperature above

            # Initialize weights with conservative bias
            # [IMPROVED] Make initial gate values more conservative
            # Find the last Linear layer (before Sigmoid) and set negative bias
            linear_layers = [m for m in self.gating_module.modules() if isinstance(m, nn.Linear)]
            for idx, m in enumerate(linear_layers):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if idx == len(linear_layers) - 1:  # Last Linear layer (before Sigmoid)
                        # Set negative bias to make initial gate values smaller
                        nn.init.constant_(m.bias, -1.0)  # Initial gate ≈ 0.27
                    else:
                        nn.init.constant_(m.bias, 0)

    # [DEBUG ADDITION]
    def visualize_reference_points(self, reference_points, layer_idx, img_metas=None, feature_map=None, feature_map_view=None, spatial_shapes=None, gt_bboxes_3d=None, gate_values=None, fused_feature=None, view_mask=None, input_lidar_img=None, input_view_img=None):
        """
        reference_points: (BS, Num_Query, 2)
        feature_map: (BS, Len_Seq, C)  Lidar
        feature_map_view: (BS, Len_Seq, C) View
        spatial_shapes: (Num_Levels, 2)
        input_lidar_img: (BS, 3, H, W) Original input lidar image
        input_view_img: (BS, 3, H, W) Original input view image
        """
        # Save every 1000 steps
        if self.debug_step % 1000 != 0:
            return

        try:
            # --- 1. Reference Points (Scatter) ---
            pts = reference_points[0].detach().cpu().numpy()
            
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
                            # Fixed size for visualization
                            img_h, img_w = 200, 200
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
            # Determine number of subplots: 3 base + 2 input images if available
            num_subplots = 3
            has_input_lidar = input_lidar_img is not None
            has_input_view = input_view_img is not None
            
            # [DEBUG] Check if input images are available
            if self.debug_step % 1000 == 0 and layer_idx == 0:
                print(f"[DEBUG VIS] Step {self.debug_step} Layer {layer_idx}: "
                      f"input_lidar_img={'available' if has_input_lidar else 'None'}, "
                      f"input_view_img={'available' if has_input_view else 'None'}")
            
            if has_input_lidar or has_input_view:
                num_subplots += 1 if (has_input_lidar and has_input_view) else 1
            
            plt.figure(figsize=(6*num_subplots, 6))
            
            plot_idx = 0
            
            # Subplot 0: Original Input Images (if available)
            if has_input_lidar or has_input_view:
                plt.subplot(1, num_subplots, plot_idx + 1)
                if has_input_lidar and has_input_view:
                    # Show both side by side
                    lidar_np = input_lidar_img[0].permute(1, 2, 0).detach().cpu().numpy()
                    view_np = input_view_img[0].permute(1, 2, 0).detach().cpu().numpy()
                    # Normalize to [0, 1]
                    lidar_np = (lidar_np - lidar_np.min()) / (lidar_np.max() - lidar_np.min() + 1e-9)
                    view_np = (view_np - view_np.min()) / (view_np.max() - view_np.min() + 1e-9)
                    # Concatenate horizontally
                    combined = np.concatenate([lidar_np, view_np], axis=1)
                    plt.imshow(combined, extent=[0, 2, 1, 0])
                    plt.axvline(x=1, color='white', linewidth=2, linestyle='--')
                    plt.text(0.5, 0.05, 'Lidar Input', ha='center', transform=plt.gca().transAxes, 
                            color='white', fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    plt.text(1.5, 0.05, 'View Input', ha='center', transform=plt.gca().transAxes, 
                            color='white', fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                elif has_input_lidar:
                    lidar_np = input_lidar_img[0].permute(1, 2, 0).detach().cpu().numpy()
                    lidar_np = (lidar_np - lidar_np.min()) / (lidar_np.max() - lidar_np.min() + 1e-9)
                    plt.imshow(lidar_np, extent=[0, 1, 1, 0])
                    plt.text(0.5, 0.05, 'Lidar Input', ha='center', transform=plt.gca().transAxes, 
                            color='white', fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                elif has_input_view:
                    view_np = input_view_img[0].permute(1, 2, 0).detach().cpu().numpy()
                    view_np = (view_np - view_np.min()) / (view_np.max() - view_np.min() + 1e-9)
                    plt.imshow(view_np, extent=[0, 1, 1, 0])
                    plt.text(0.5, 0.05, 'View Input', ha='center', transform=plt.gca().transAxes, 
                            color='white', fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                plt.xlim(0, 2 if (has_input_lidar and has_input_view) else 1)
                plt.ylim(1, 0)
                plt.title("Original Input Images")
                plt.axis('off')
                plot_idx += 1
            
            # Subplot 1: Lidar Feature Map + Scatter Points
            plt.subplot(1, num_subplots, plot_idx + 1)
            plot_idx += 1  # [FIX] Update plot_idx after creating subplot
            if feat_img_lidar is not None:
                # Normal Colors
                plt.imshow(feat_img_lidar, cmap='viridis', extent=[0, 1, 1, 0])
                # Scatter points (red)
                plt.scatter(pts[:, 0], pts[:, 1], c='red', s=5, alpha=0.9, edgecolors='white', linewidths=0.1, label='Pred')
                
                # Draw GT if available (Green)
                gt_pts = None
                if gt_bboxes_3d is not None:
                    try:
                        gt = gt_bboxes_3d[0] # First sample
                        
                        if hasattr(gt, 'instance_list'):
                             all_pts = []
                             for line in gt.instance_list:
                                 l_pts = np.array(line.coords)
                                 # Normalize if needed
                                 if hasattr(gt, 'max_x') and hasattr(gt, 'max_y'):
                                      l_pts[:, 0] /= gt.max_x
                                      l_pts[:, 1] /= gt.max_y
                                 all_pts.append(l_pts)
                             if len(all_pts) > 0:
                                 gt_pts = np.concatenate(all_pts, axis=0)

                        elif hasattr(gt, 'tensor'):
                            gt_pts = gt.tensor.cpu().numpy()
                        else:
                            gt_pts = gt.cpu().numpy()
                        
                        if gt_pts is not None:
                            if gt_pts.ndim == 3:
                                gt_pts = gt_pts.reshape(-1, 2)
                            
                            # GT usually dense, use smaller size
                            plt.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lime', s=2, alpha=0.6, label='GT')
                            
                    except Exception as e:
                        print(f"GT Vis Error: {e}")
                
                # Add Legend
                plt.legend(loc='upper right', fontsize='small')
                plt.xlim(0, 1)
                plt.ylim(1, 0) # Inverted Y for image coords
                plt.title("Lidar (Normal) + Pred(Red) + GT(Green)")
            else:
                plt.text(0.5, 0.5, status_lidar or "No Lidar", ha='center')
                plt.title("Lidar Skipped")

            # Subplot 2: View Feature Map + Scatter Points
            plt.subplot(1, num_subplots, plot_idx + 1)
            plot_idx += 1  # [FIX] Update plot_idx after creating subplot
            if feat_img_view is not None:
                plt.imshow(feat_img_view, cmap='viridis', extent=[0, 1, 1, 0])
                plt.scatter(pts[:, 0], pts[:, 1], c='red', s=5, alpha=0.9, edgecolors='white', linewidths=0.1)
                if gt_pts is not None:
                     plt.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lime', s=2, alpha=0.6)
                plt.xlim(0, 1)
                plt.ylim(1, 0)
                plt.title("View (Normal) + Pred + GT")
            else:
                plt.text(0.5, 0.5, status_view or "No View", ha='center')

            # Subplot 3: Combined Overlay (To check alignment)
            plt.subplot(1, num_subplots, plot_idx + 1)
            if feat_img_lidar is not None and feat_img_view is not None:
                # Blend Lidar (Green Channel) and View (Red Channel) to see misalignment
                H, W = feat_img_lidar.shape
                blend = np.zeros((H, W, 3))
                blend[..., 0] = feat_img_view    # Red = View
                blend[..., 1] = feat_img_lidar   # Green = Lidar
                # Blue = 0
                
                plt.imshow(blend, extent=[0, 1, 1, 0])
                plt.scatter(pts[:, 0], pts[:, 1], c='white', s=3, alpha=0.8, label='Pred')
                plt.title("Alignment Check: R=View, G=Lidar")
            else: 
                plt.text(0.5, 0.5, "Cannot Blend", ha='center')

            plt.suptitle(f"Step {self.debug_step} Layer {layer_idx}")
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_scatter.png")
            plt.savefig(save_path)
            plt.close()
            
            # [ENHANCED DEBUG] Additional diagnostic plots
            if gate_values is not None or fused_feature is not None or view_mask is not None:
                self.visualize_fusion_diagnostics(
                    reference_points, layer_idx, 
                    feature_map, feature_map_view, fused_feature,
                    gate_values, view_mask, spatial_shapes, gt_bboxes_3d
                )
            
        except Exception as e:
            print(f"Vis error: {e}")

    def visualize_fusion_diagnostics(self, reference_points, layer_idx, 
                                     feature_map_lidar, feature_map_view, fused_feature,
                                     gate_values, view_mask, spatial_shapes, gt_bboxes_3d):
        """Enhanced visualization for fusion diagnostics"""
        if self.debug_step % 1000 != 0:
            return
            
        try:
            pts = reference_points[0].detach().cpu().numpy()
            
            def process_feat(feat_map, s_shapes):
                feat_img = None
                status = ""
                if feat_map is not None and s_shapes is not None:
                    try:
                        # Handle different input shapes
                        if feat_map.dim() == 2:
                            # (BS, Num_Keys) or (Num_Keys, C)
                            if feat_map.shape[0] > feat_map.shape[1] and feat_map.shape[1] > 1:
                                seq_dim = 0
                            else:
                                seq_dim = 1
                        elif feat_map.dim() == 3:
                            # (BS, Num_Keys, C) or (Num_Keys, BS, C)
                            if feat_map.shape[1] != feat_map.shape[0] and feat_map.shape[0] > feat_map.shape[1]:
                                seq_dim = 0
                            else:
                                seq_dim = 1
                        else:
                            status = f"Unexpected dim: {feat_map.dim()}"
                            return feat_img, status
                        
                        current_len = feat_map.shape[seq_dim]
                        H, W = s_shapes[0].tolist()
                        feat_len = H * W
                        
                        if current_len >= feat_len:
                            if seq_dim == 0:
                                if feat_map.dim() == 2:
                                    feat = feat_map[:feat_len, :].detach().cpu()
                                else:
                                    feat = feat_map[:feat_len, 0, :].detach().cpu()
                            else:
                                if feat_map.dim() == 2:
                                    feat = feat_map[0, :feat_len].unsqueeze(-1).detach().cpu()  # Add channel dim
                                else:
                                    feat = feat_map[0, :feat_len, :].detach().cpu()
                            
                            # Compute norm if multi-channel, otherwise use directly
                            if feat.dim() == 2 and feat.shape[1] > 1:
                                feat_norm = torch.norm(feat, dim=1).numpy()
                            else:
                                feat_norm = feat.squeeze(-1).numpy() if feat.dim() == 2 else feat.numpy()
                            
                            try:
                                img_h, img_w = 200, 200
                                feat_img = feat_norm.reshape(H, W)
                                feat_img = cv2.resize(feat_img, (img_w, img_h))
                                feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min() + 1e-9)
                            except (ValueError, RuntimeError) as e:
                                status = f"Reshape/Resize error: {e}"
                        else:
                            status = f"Size mismatch: {current_len} < {feat_len}"
                    except Exception as e:
                        status = f"Process error: {e}"
                else:
                    status = "No input"
                return feat_img, status
            
            # 动态确定需要的子图数量，避免出现未使用的空白坐标轴
            num_plots = 0
            has_gate_plot = gate_values is not None
            has_view_mask_plot = view_mask is not None
            has_fused_plot = fused_feature is not None

            if has_gate_plot:
                num_plots += 1
            if has_view_mask_plot:
                num_plots += 1
            if has_fused_plot:
                num_plots += 1

            # 统计信息单独占一个子图（只要有任意一种特征存在就画）
            has_stats_plot = (feature_map_lidar is not None) or \
                             (feature_map_view is not None) or \
                             (fused_feature is not None)
            if has_stats_plot:
                num_plots += 1
                
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
            if num_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # Plot 1: Gate values
            if gate_values is not None:
                try:
                    # gate_values: (BS, Num_Keys, C) -> take mean over channel dim
                    gate_mean = gate_values.mean(dim=-1)  # (BS, Num_Keys)
                    if gate_mean.dim() == 2:
                        # Add channel dim for process_feat: (BS, Num_Keys) -> (BS, Num_Keys, 1)
                        gate_mean = gate_mean.unsqueeze(-1)
                    gate_img, gate_status = process_feat(gate_mean, spatial_shapes)
                    if gate_img is not None:
                        axes[plot_idx].imshow(gate_img, cmap='hot', extent=[0, 1, 1, 0])
                        axes[plot_idx].scatter(pts[:, 0], pts[:, 1], c='cyan', s=3, alpha=0.6)
                        axes[plot_idx].set_title(f"Gate Values (mean={gate_values.mean().item():.3f})")
                        axes[plot_idx].set_xlim(0, 1)
                        axes[plot_idx].set_ylim(1, 0)
                    else:
                        axes[plot_idx].text(0.5, 0.5, f"Gate: {gate_status}", ha='center', va='center')
                        axes[plot_idx].set_title("Gate Values (Failed)")
                    plot_idx += 1
                except Exception as e:
                    if plot_idx < len(axes):
                        axes[plot_idx].text(0.5, 0.5, f"Gate Error: {e}", ha='center', va='center')
                        plot_idx += 1
            
            # Plot 2: View Mask
            if view_mask is not None:
                try:
                    # view_mask: (BS, Num_Keys) bool, True=invalid, False=valid
                    # Convert to float for visualization: 1.0=invalid, 0.0=valid
                    view_mask_float = view_mask.float() if view_mask.dtype == torch.bool else view_mask
                    if view_mask_float.dim() == 2:
                        view_mask_expanded = view_mask_float.unsqueeze(-1)  # (BS, Num_Keys, 1)
                    else:
                        view_mask_expanded = view_mask_float
                    mask_img, mask_status = process_feat(view_mask_expanded, spatial_shapes)
                    if mask_img is not None:
                        axes[plot_idx].imshow(mask_img, cmap='gray', extent=[0, 1, 1, 0])
                        # valid_ratio: False (valid) / total
                        valid_ratio = (~view_mask).float().mean().item() if view_mask.dtype == torch.bool else (view_mask < 0.5).float().mean().item()
                        axes[plot_idx].set_title(f"View Mask (valid={valid_ratio:.2%})")
                        axes[plot_idx].set_xlim(0, 1)
                        axes[plot_idx].set_ylim(1, 0)
                    else:
                        axes[plot_idx].text(0.5, 0.5, f"Mask: {mask_status}", ha='center', va='center')
                        axes[plot_idx].set_title("View Mask (Failed)")
                    plot_idx += 1
                except Exception as e:
                    if plot_idx < len(axes):
                        axes[plot_idx].text(0.5, 0.5, f"Mask Error: {e}", ha='center', va='center')
                        plot_idx += 1
            
            # Plot 3: Fused Feature
            if fused_feature is not None:
                try:
                    fused_img, fused_status = process_feat(fused_feature, spatial_shapes)
                    if fused_img is not None:
                        axes[plot_idx].imshow(fused_img, cmap='viridis', extent=[0, 1, 1, 0])
                        axes[plot_idx].scatter(pts[:, 0], pts[:, 1], c='red', s=3, alpha=0.6)
                        axes[plot_idx].set_title("Fused Feature")
                        axes[plot_idx].set_xlim(0, 1)
                        axes[plot_idx].set_ylim(1, 0)
                    else:
                        axes[plot_idx].text(0.5, 0.5, f"Fused: {fused_status}", ha='center', va='center')
                        axes[plot_idx].set_title("Fused Feature (Failed)")
                    plot_idx += 1
                except Exception as e:
                    if plot_idx < len(axes):
                        axes[plot_idx].text(0.5, 0.5, f"Fused Error: {e}", ha='center', va='center')
                        plot_idx += 1
            
            # Plot 4: Feature Statistics
            stats_text = []
            if feature_map_lidar is not None:
                lidar_mean = feature_map_lidar.mean().item()
                lidar_std = feature_map_lidar.std().item()
                stats_text.append(f"Lidar: μ={lidar_mean:.3f}, σ={lidar_std:.3f}")
            if feature_map_view is not None:
                view_mean = feature_map_view.mean().item()
                view_std = feature_map_view.std().item()
                stats_text.append(f"View: μ={view_mean:.3f}, σ={view_std:.3f}")
            if fused_feature is not None:
                fused_mean = fused_feature.mean().item()
                fused_std = fused_feature.std().item()
                stats_text.append(f"Fused: μ={fused_mean:.3f}, σ={fused_std:.3f}")
            
            if len(stats_text) > 0 and plot_idx < len(axes):
                axes[plot_idx].text(0.1, 0.5, '\n'.join(stats_text), 
                                   transform=axes[plot_idx].transAxes,
                                   fontsize=10, verticalalignment='center',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[plot_idx].set_title("Feature Statistics")
                axes[plot_idx].axis('off')
            
            plt.suptitle(f"Fusion Diagnostics - Step {self.debug_step} Layer {layer_idx}")
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_fusion_diag.png")
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Fusion diag vis error: {e}")

    def visualize_adaptive_weights(self, lidar_weight, view_weight, reference_points, layer_idx, 
                                   gt_bboxes_3d=None, spatial_shapes=None):
        """
        可视化每个query的自适应融合权重
        
        Args:
            lidar_weight: (bs, num_query, 1) Lidar融合权重
            view_weight: (bs, num_query, 1) View融合权重
            reference_points: (bs, num_query, 2) Reference points
            layer_idx: 当前层索引
            gt_bboxes_3d: Ground truth boxes (可选)
            spatial_shapes: Spatial shapes for BEV visualization (可选)
        """
        if self.debug_step % 1000 != 0:
            return
        
        try:
            # 提取权重
            weights_lidar = lidar_weight[0, :, 0].detach().cpu().numpy()
            weights_view = view_weight[0, :, 0].detach().cpu().numpy()
            pts = reference_points[0].detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1. 权重直方图
            axes[0].hist(weights_lidar, bins=50, alpha=0.7, label='Lidar', color='green', edgecolor='black')
            axes[0].hist(weights_view, bins=50, alpha=0.7, label='View', color='red', edgecolor='black')
            axes[0].set_xlabel('Weight Value', fontsize=12)
            axes[0].set_ylabel('Query Count', fontsize=12)
            axes[0].set_title(f'Weight Distribution (Layer {layer_idx})', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # 添加统计信息
            lidar_mean = weights_lidar.mean()
            view_mean = weights_view.mean()
            axes[0].axvline(lidar_mean, color='green', linestyle='--', linewidth=2, label=f'Lidar mean={lidar_mean:.3f}')
            axes[0].axvline(view_mean, color='red', linestyle='--', linewidth=2, label=f'View mean={view_mean:.3f}')
            
            # 2. 权重散点图 (Lidar vs View)
            axes[1].scatter(weights_lidar, weights_view, alpha=0.5, s=10, c='blue', edgecolors='black', linewidths=0.1)
            axes[1].plot([0, 1], [1, 0], 'r--', linewidth=2, label='sum=1', alpha=0.7)
            axes[1].set_xlabel('Lidar Weight', fontsize=12)
            axes[1].set_ylabel('View Weight', fontsize=12)
            axes[1].set_title('Lidar vs View Weights (Per Query)', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            
            # 添加对角线标注
            axes[1].text(0.5, 0.5, 'Equal weights', rotation=-45, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3), fontsize=10)
            
            # 3. 权重空间分布（在BEV空间上可视化）
            if spatial_shapes is not None:
                H, W = spatial_shapes[0].tolist()
                # 创建BEV权重图
                lidar_weight_map = np.zeros((H, W))
                view_weight_map = np.zeros((H, W))
                
                # 将reference points映射到BEV空间
                for i, (x, y) in enumerate(pts):
                    x_idx = int(x * W)
                    y_idx = int(y * H)
                    x_idx = np.clip(x_idx, 0, W - 1)
                    y_idx = np.clip(y_idx, 0, H - 1)
                    lidar_weight_map[y_idx, x_idx] = weights_lidar[i]
                    view_weight_map[y_idx, x_idx] = weights_view[i]
                
                # 使用双通道显示：绿色=Lidar权重，红色=View权重
                weight_blend = np.zeros((H, W, 3))
                weight_blend[..., 0] = view_weight_map  # Red = View
                weight_blend[..., 1] = lidar_weight_map  # Green = Lidar
                # Blue = 0
                
                # Resize for visualization
                img_h, img_w = 200, 200
                weight_blend_resized = cv2.resize(weight_blend, (img_w, img_h))
                weight_blend_resized = np.clip(weight_blend_resized, 0, 1)
                
                axes[2].imshow(weight_blend_resized, extent=[0, 1, 1, 0])
                axes[2].scatter(pts[:, 0], pts[:, 1], c='cyan', s=3, alpha=0.6, edgecolors='white', linewidths=0.1)
                axes[2].set_xlabel('X (Normalized)', fontsize=12)
                axes[2].set_ylabel('Y (Normalized)', fontsize=12)
                axes[2].set_title('Spatial Weight Distribution\n(R=View, G=Lidar)', fontsize=14, fontweight='bold')
                axes[2].set_xlim(0, 1)
                axes[2].set_ylim(1, 0)
            else:
                # 如果没有spatial_shapes，显示权重排序
                sorted_indices = np.argsort(weights_lidar)
                axes[2].plot(weights_lidar[sorted_indices], label='Lidar', color='green', linewidth=2)
                axes[2].plot(weights_view[sorted_indices], label='View', color='red', linewidth=2)
                axes[2].set_xlabel('Query Index (Sorted by Lidar Weight)', fontsize=12)
                axes[2].set_ylabel('Weight Value', fontsize=12)
                axes[2].set_title('Weight Distribution (Sorted)', fontsize=14, fontweight='bold')
                axes[2].legend(fontsize=11)
                axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Adaptive Fusion Weights - Step {self.debug_step} Layer {layer_idx}', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_adaptive_weights.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Adaptive weights vis error: {e}")

    def visualize_cross_attention(self, attn_weights_lidar, attn_weights_view, reference_points, 
                                  layer_idx, spatial_shapes=None, num_sample_queries=5):
        """
        可视化query对Lidar和View的attention分布
        
        Args:
            attn_weights_lidar: (bs, num_query, num_keys) 或 (bs, num_heads, num_query, num_keys)
            attn_weights_view: (bs, num_query, num_keys) 或 (bs, num_heads, num_query, num_keys)
            reference_points: (bs, num_query, 2)
            layer_idx: 当前层索引
            spatial_shapes: Spatial shapes for BEV visualization
            num_sample_queries: 选择多少个代表性的query进行可视化
        """
        if self.debug_step % 1000 != 0:
            return
        
        try:
            # 处理attention weights的维度
            if attn_weights_lidar.dim() == 4:  # (bs, num_heads, num_query, num_keys)
                # 平均所有head
                attn_weights_lidar = attn_weights_lidar.mean(dim=1)  # (bs, num_query, num_keys)
            if attn_weights_view.dim() == 4:
                attn_weights_view = attn_weights_view.mean(dim=1)
            
            bs, num_query, num_keys = attn_weights_lidar.shape
            pts = reference_points[0].detach().cpu().numpy()
            
            # 选择代表性的queries（选择attention熵最大的，即最"不确定"的）
            attn_entropy_lidar = -(attn_weights_lidar[0] * torch.log(attn_weights_lidar[0] + 1e-9)).sum(dim=-1)
            attn_entropy_view = -(attn_weights_view[0] * torch.log(attn_weights_view[0] + 1e-9)).sum(dim=-1)
            combined_entropy = attn_entropy_lidar + attn_entropy_view
            top_queries = torch.topk(combined_entropy, min(num_sample_queries, num_query)).indices.cpu().numpy()
            
            num_plots = len(top_queries)
            fig, axes = plt.subplots(num_plots, 2, figsize=(12, 4 * num_plots))
            if num_plots == 1:
                axes = axes.reshape(1, -1)
            
            for plot_idx, q_idx in enumerate(top_queries):
                # Lidar attention
                attn_lidar = attn_weights_lidar[0, q_idx].detach().cpu().numpy()
                # View attention
                attn_view = attn_weights_view[0, q_idx].detach().cpu().numpy()
                
                # 可视化Lidar attention
                if spatial_shapes is not None:
                    H, W = spatial_shapes[0].tolist()
                    # 将attention weights reshape到BEV空间
                    if len(attn_lidar) >= H * W:
                        attn_lidar_map = attn_lidar[:H*W].reshape(H, W)
                        attn_view_map = attn_view[:H*W].reshape(H, W)
                        
                        # Resize for visualization
                        img_h, img_w = 200, 200
                        attn_lidar_resized = cv2.resize(attn_lidar_map, (img_w, img_h))
                        attn_view_resized = cv2.resize(attn_view_map, (img_w, img_h))
                        
                        axes[plot_idx, 0].imshow(attn_lidar_resized, cmap='hot', extent=[0, 1, 1, 0])
                        axes[plot_idx, 0].scatter(pts[q_idx, 0], pts[q_idx, 1], c='cyan', s=50, 
                                                  marker='*', edgecolors='white', linewidths=1)
                        axes[plot_idx, 0].set_title(f'Query {q_idx}: Lidar Attention', fontweight='bold')
                        axes[plot_idx, 0].set_xlim(0, 1)
                        axes[plot_idx, 0].set_ylim(1, 0)
                        
                        axes[plot_idx, 1].imshow(attn_view_resized, cmap='hot', extent=[0, 1, 1, 0])
                        axes[plot_idx, 1].scatter(pts[q_idx, 0], pts[q_idx, 1], c='cyan', s=50,
                                                  marker='*', edgecolors='white', linewidths=1)
                        axes[plot_idx, 1].set_title(f'Query {q_idx}: View Attention', fontweight='bold')
                        axes[plot_idx, 1].set_xlim(0, 1)
                        axes[plot_idx, 1].set_ylim(1, 0)
                    else:
                        # 如果无法reshape，显示attention分布
                        axes[plot_idx, 0].bar(range(len(attn_lidar)), attn_lidar, alpha=0.7, color='green')
                        axes[plot_idx, 0].set_title(f'Query {q_idx}: Lidar Attention Distribution')
                        axes[plot_idx, 1].bar(range(len(attn_view)), attn_view, alpha=0.7, color='red')
                        axes[plot_idx, 1].set_title(f'Query {q_idx}: View Attention Distribution')
                else:
                    # 没有spatial_shapes，显示attention分布
                    axes[plot_idx, 0].bar(range(len(attn_lidar)), attn_lidar, alpha=0.7, color='green')
                    axes[plot_idx, 0].set_title(f'Query {q_idx}: Lidar Attention Distribution')
                    axes[plot_idx, 1].bar(range(len(attn_view)), attn_view, alpha=0.7, color='red')
                    axes[plot_idx, 1].set_title(f'Query {q_idx}: View Attention Distribution')
            
            plt.suptitle(f'Cross-Attention Visualization - Step {self.debug_step} Layer {layer_idx}',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_cross_attention.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Cross-attention vis error: {e}")

    def visualize_feature_comparison(self, feat_lidar, feat_view, feat_fused, reference_points,
                                     layer_idx, gt_bboxes_3d=None, spatial_shapes=None):
        """
        对比融合前后的特征质量
        
        Args:
            feat_lidar: (bs, num_query, embed_dims) Lidar特征
            feat_view: (bs, num_query, embed_dims) View特征
            feat_fused: (bs, num_query, embed_dims) 融合后特征
            reference_points: (bs, num_query, 2)
            layer_idx: 当前层索引
            gt_bboxes_3d: Ground truth (可选)
            spatial_shapes: Spatial shapes (可选)
        """
        if self.debug_step % 1000 != 0:
            return
        
        try:
            # 计算特征统计信息
            lidar_norm = torch.norm(feat_lidar[0], dim=-1).detach().cpu().numpy()
            view_norm = torch.norm(feat_view[0], dim=-1).detach().cpu().numpy()
            fused_norm = torch.norm(feat_fused[0], dim=-1).detach().cpu().numpy()
            
            # 计算特征与GT的相关性（如果有GT）
            gt_correlation = None
            if gt_bboxes_3d is not None:
                try:
                    gt = gt_bboxes_3d[0]
                    if hasattr(gt, 'instance_list'):
                        # 计算预测点与GT的距离
                        all_gt_pts = []
                        for line in gt.instance_list:
                            l_pts = np.array(line.coords)
                            if hasattr(gt, 'max_x') and hasattr(gt, 'max_y'):
                                l_pts[:, 0] /= gt.max_x
                                l_pts[:, 1] /= gt.max_y
                            all_gt_pts.append(l_pts)
                        if len(all_gt_pts) > 0:
                            gt_pts = np.concatenate(all_gt_pts, axis=0)
                            pts = reference_points[0].detach().cpu().numpy()
                            
                            # 计算每个query到最近GT点的距离
                            if cdist is not None:
                                distances = cdist(pts, gt_pts)
                                min_distances = distances.min(axis=1)
                                gt_correlation = min_distances
                            else:
                                # 如果scipy不可用，使用简单的欧氏距离
                                min_distances = []
                                for pt in pts:
                                    dists = np.sqrt(((gt_pts - pt) ** 2).sum(axis=1))
                                    min_distances.append(dists.min())
                                gt_correlation = np.array(min_distances)
                except Exception as e:
                    print(f"GT correlation calculation error: {e}")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # 第一行：特征幅度分布
            axes[0, 0].hist(lidar_norm, bins=50, alpha=0.7, label='Lidar', color='green', edgecolor='black')
            axes[0, 0].hist(view_norm, bins=50, alpha=0.7, label='View', color='red', edgecolor='black')
            axes[0, 0].hist(fused_norm, bins=50, alpha=0.7, label='Fused', color='blue', edgecolor='black')
            axes[0, 0].set_xlabel('Feature Norm', fontsize=12)
            axes[0, 0].set_ylabel('Count', fontsize=12)
            axes[0, 0].set_title('Feature Magnitude Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].legend(fontsize=11)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 第二行第一列：特征幅度对比（散点图）
            axes[0, 1].scatter(lidar_norm, view_norm, alpha=0.5, s=10, c='purple', edgecolors='black', linewidths=0.1)
            axes[0, 1].plot([0, lidar_norm.max()], [0, view_norm.max()], 'r--', linewidth=2, alpha=0.7, label='y=x')
            axes[0, 1].set_xlabel('Lidar Feature Norm', fontsize=12)
            axes[0, 1].set_ylabel('View Feature Norm', fontsize=12)
            axes[0, 1].set_title('Lidar vs View Feature Magnitude', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=11)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 第一行第三列：融合前后对比
            axes[0, 2].scatter(lidar_norm, fused_norm, alpha=0.5, s=10, c='green', label='Lidar→Fused', 
                              edgecolors='black', linewidths=0.1)
            axes[0, 2].scatter(view_norm, fused_norm, alpha=0.5, s=10, c='red', label='View→Fused',
                              edgecolors='black', linewidths=0.1)
            axes[0, 2].plot([0, max(lidar_norm.max(), view_norm.max())], 
                           [0, max(lidar_norm.max(), view_norm.max())], 'b--', linewidth=2, alpha=0.7)
            axes[0, 2].set_xlabel('Input Feature Norm', fontsize=12)
            axes[0, 2].set_ylabel('Fused Feature Norm', fontsize=12)
            axes[0, 2].set_title('Fusion Effect: Input vs Output', fontsize=14, fontweight='bold')
            axes[0, 2].legend(fontsize=11)
            axes[0, 2].grid(True, alpha=0.3)
            
            # 第二行：如果有GT，显示特征与GT的相关性
            if gt_correlation is not None:
                axes[1, 0].scatter(gt_correlation, lidar_norm, alpha=0.5, s=10, c='green', 
                                  label='Lidar', edgecolors='black', linewidths=0.1)
                axes[1, 0].scatter(gt_correlation, view_norm, alpha=0.5, s=10, c='red',
                                  label='View', edgecolors='black', linewidths=0.1)
                axes[1, 0].scatter(gt_correlation, fused_norm, alpha=0.5, s=10, c='blue',
                                  label='Fused', edgecolors='black', linewidths=0.1)
                axes[1, 0].set_xlabel('Distance to GT (Normalized)', fontsize=12)
                axes[1, 0].set_ylabel('Feature Norm', fontsize=12)
                axes[1, 0].set_title('Feature Quality vs GT Distance', fontsize=14, fontweight='bold')
                axes[1, 0].legend(fontsize=11)
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No GT available', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=14)
                axes[1, 0].set_title('Feature Quality vs GT Distance', fontsize=14, fontweight='bold')
            
            # 统计信息文本
            stats_text = [
                f"Lidar: μ={lidar_norm.mean():.3f}, σ={lidar_norm.std():.3f}",
                f"View: μ={view_norm.mean():.3f}, σ={view_norm.std():.3f}",
                f"Fused: μ={fused_norm.mean():.3f}, σ={fused_norm.std():.3f}",
                f"Fusion Gain: {(fused_norm.mean() / (lidar_norm.mean() + view_norm.mean()) * 2):.3f}"
            ]
            axes[1, 1].text(0.1, 0.5, '\n'.join(stats_text), transform=axes[1, 1].transAxes,
                          fontsize=12, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Feature Statistics', fontsize=14, fontweight='bold')
            
            # 特征空间分布（如果有spatial_shapes）
            if spatial_shapes is not None:
                H, W = spatial_shapes[0].tolist()
                pts = reference_points[0].detach().cpu().numpy()
                
                # 创建BEV特征图
                def create_feat_map(feat_norm, pts, H, W):
                    feat_map = np.zeros((H, W))
                    for i, (x, y) in enumerate(pts):
                        x_idx = int(x * W)
                        y_idx = int(y * H)
                        x_idx = np.clip(x_idx, 0, W - 1)
                        y_idx = np.clip(y_idx, 0, H - 1)
                        feat_map[y_idx, x_idx] = feat_norm[i]
                    return feat_map
                
                lidar_map = create_feat_map(lidar_norm, pts, H, W)
                view_map = create_feat_map(view_norm, pts, H, W)
                fused_map = create_feat_map(fused_norm, pts, H, W)
                
                # Resize
                img_h, img_w = 200, 200
                lidar_resized = cv2.resize(lidar_map, (img_w, img_h))
                view_resized = cv2.resize(view_map, (img_w, img_h))
                fused_resized = cv2.resize(fused_map, (img_w, img_h))
                
                # 归一化
                lidar_resized = (lidar_resized - lidar_resized.min()) / (lidar_resized.max() - lidar_resized.min() + 1e-9)
                view_resized = (view_resized - view_resized.min()) / (view_resized.max() - view_resized.min() + 1e-9)
                fused_resized = (fused_resized - fused_resized.min()) / (fused_resized.max() - fused_resized.min() + 1e-9)
                
                # 显示融合后的特征图
                axes[1, 2].imshow(fused_resized, cmap='viridis', extent=[0, 1, 1, 0])
                axes[1, 2].scatter(pts[:, 0], pts[:, 1], c='red', s=3, alpha=0.6, edgecolors='white', linewidths=0.1)
                axes[1, 2].set_xlabel('X (Normalized)', fontsize=12)
                axes[1, 2].set_ylabel('Y (Normalized)', fontsize=12)
                axes[1, 2].set_title('Fused Feature Spatial Distribution', fontsize=14, fontweight='bold')
                axes[1, 2].set_xlim(0, 1)
                axes[1, 2].set_ylim(1, 0)
            else:
                axes[1, 2].axis('off')
            
            plt.suptitle(f'Feature Comparison - Step {self.debug_step} Layer {layer_idx}',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_feature_comparison.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Feature comparison vis error: {e}")

    def visualize_layer_progression(self, all_layer_outputs, all_reference_points, all_weights, gt_bboxes_3d=None):
        """
        追踪6层decoder中融合效果的变化
        
        Args:
            all_layer_outputs: List of (bs, num_query, embed_dims) 每层的输出
            all_reference_points: List of (bs, num_query, 2) 每层的reference points
            all_weights: List of (lidar_weight, view_weight) tuples 每层的融合权重
            gt_bboxes_3d: Ground truth (可选)
        """
        if self.debug_step % 1000 != 0:
            return
        
        try:
            num_layers = len(all_layer_outputs)
            if num_layers == 0:
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            # 计算每层的特征统计和与GT的距离
            layer_stats = []
            for lid in range(num_layers):
                output = all_layer_outputs[lid]
                ref_pts = all_reference_points[lid]
                
                # 特征统计
                feat_norm = torch.norm(output[0], dim=-1).detach().cpu().numpy()
                mean_norm = feat_norm.mean()
                std_norm = feat_norm.std()
                
                # 计算与GT的距离（如果有GT）
                gt_distance = None
                if gt_bboxes_3d is not None:
                    try:
                        gt = gt_bboxes_3d[0]
                        if hasattr(gt, 'instance_list'):
                            all_gt_pts = []
                            for line in gt.instance_list:
                                l_pts = np.array(line.coords)
                                if hasattr(gt, 'max_x') and hasattr(gt, 'max_y'):
                                    l_pts[:, 0] /= gt.max_x
                                    l_pts[:, 1] /= gt.max_y
                                all_gt_pts.append(l_pts)
                            if len(all_gt_pts) > 0:
                                gt_pts = np.concatenate(all_gt_pts, axis=0)
                                pts = ref_pts[0].detach().cpu().numpy()
                                if cdist is not None:
                                    distances = cdist(pts, gt_pts)
                                    gt_distance = distances.min(axis=1).mean()
                                else:
                                    # 如果scipy不可用，使用简单的欧氏距离
                                    min_distances = []
                                    for pt in pts:
                                        dists = np.sqrt(((gt_pts - pt) ** 2).sum(axis=1))
                                        min_distances.append(dists.min())
                                    gt_distance = np.array(min_distances).mean()
                    except Exception:
                        pass
                
                # 融合权重统计
                if lid < len(all_weights) and all_weights[lid] is not None:
                    lidar_w, view_w = all_weights[lid]
                    if isinstance(lidar_w, torch.Tensor):
                        lidar_w_mean = lidar_w[0, :, 0].mean().item() if lidar_w.dim() == 3 else lidar_w.mean().item()
                        view_w_mean = view_w[0, :, 0].mean().item() if view_w.dim() == 3 else view_w.mean().item()
                    else:
                        lidar_w_mean = lidar_w
                        view_w_mean = view_w
                else:
                    lidar_w_mean = 0.5
                    view_w_mean = 0.5
                
                layer_stats.append({
                    'layer': lid,
                    'mean_norm': mean_norm,
                    'std_norm': std_norm,
                    'gt_distance': gt_distance,
                    'lidar_weight': lidar_w_mean,
                    'view_weight': view_w_mean
                })
            
            # 1. 特征幅度随层数变化
            layers = [s['layer'] for s in layer_stats]
            mean_norms = [s['mean_norm'] for s in layer_stats]
            std_norms = [s['std_norm'] for s in layer_stats]
            
            axes[0].plot(layers, mean_norms, 'o-', linewidth=2, markersize=8, label='Mean Norm', color='blue')
            axes[0].fill_between(layers, 
                                [m - s for m, s in zip(mean_norms, std_norms)],
                                [m + s for m, s in zip(mean_norms, std_norms)],
                                alpha=0.3, color='blue', label='±1 std')
            axes[0].set_xlabel('Layer Index', fontsize=12)
            axes[0].set_ylabel('Feature Norm', fontsize=12)
            axes[0].set_title('Feature Magnitude Progression', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xticks(layers)
            
            # 2. 融合权重随层数变化
            lidar_weights = [s['lidar_weight'] for s in layer_stats]
            view_weights = [s['view_weight'] for s in layer_stats]
            
            axes[1].plot(layers, lidar_weights, 'o-', linewidth=2, markersize=8, label='Lidar Weight', color='green')
            axes[1].plot(layers, view_weights, 's-', linewidth=2, markersize=8, label='View Weight', color='red')
            axes[1].set_xlabel('Layer Index', fontsize=12)
            axes[1].set_ylabel('Fusion Weight', fontsize=12)
            axes[1].set_title('Fusion Weight Progression', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
            axes[1].set_xticks(layers)
            
            # 3. 与GT的距离随层数变化（如果有GT）
            if any(s['gt_distance'] is not None for s in layer_stats):
                gt_distances = [s['gt_distance'] if s['gt_distance'] is not None else 0 for s in layer_stats]
                axes[2].plot(layers, gt_distances, 'o-', linewidth=2, markersize=8, label='Mean Distance to GT', color='purple')
                axes[2].set_xlabel('Layer Index', fontsize=12)
                axes[2].set_ylabel('Distance to GT (Normalized)', fontsize=12)
                axes[2].set_title('Prediction Accuracy Progression', fontsize=14, fontweight='bold')
                axes[2].legend(fontsize=11)
                axes[2].grid(True, alpha=0.3)
                axes[2].set_xticks(layers)
            else:
                axes[2].text(0.5, 0.5, 'No GT available', ha='center', va='center',
                           transform=axes[2].transAxes, fontsize=14)
                axes[2].set_title('Prediction Accuracy Progression', fontsize=14, fontweight='bold')
            
            # 4. 权重变化趋势（Lidar vs View）
            axes[3].scatter(lidar_weights, view_weights, s=100, c=layers, cmap='viridis', 
                           edgecolors='black', linewidths=2, alpha=0.7)
            axes[3].plot([0, 1], [1, 0], 'r--', linewidth=2, alpha=0.5, label='sum=1')
            for i, lid in enumerate(layers):
                axes[3].annotate(f'L{lid}', (lidar_weights[i], view_weights[i]), 
                               fontsize=10, ha='center', va='center', color='white', fontweight='bold')
            axes[3].set_xlabel('Lidar Weight', fontsize=12)
            axes[3].set_ylabel('View Weight', fontsize=12)
            axes[3].set_title('Weight Evolution Across Layers', fontsize=14, fontweight='bold')
            axes[3].legend(fontsize=11)
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlim(0, 1)
            axes[3].set_ylim(0, 1)
            
            # 5. 特征质量指标（综合）
            # 计算"质量分数"：特征幅度 / (1 + GT距离)
            quality_scores = []
            for s in layer_stats:
                if s['gt_distance'] is not None:
                    quality = s['mean_norm'] / (1 + s['gt_distance'])
                else:
                    quality = s['mean_norm']
                quality_scores.append(quality)
            
            axes[4].plot(layers, quality_scores, 'o-', linewidth=2, markersize=8, label='Quality Score', color='orange')
            axes[4].set_xlabel('Layer Index', fontsize=12)
            axes[4].set_ylabel('Quality Score', fontsize=12)
            axes[4].set_title('Overall Feature Quality Progression', fontsize=14, fontweight='bold')
            axes[4].legend(fontsize=11)
            axes[4].grid(True, alpha=0.3)
            axes[4].set_xticks(layers)
            
            # 6. 统计信息表格
            stats_text = ["Layer Progression Summary:\n"]
            for s in layer_stats:
                stats_text.append(f"L{s['layer']}: Norm={s['mean_norm']:.3f}, "
                                f"Lidar={s['lidar_weight']:.2f}, View={s['view_weight']:.2f}")
                if s['gt_distance'] is not None:
                    stats_text[-1] += f", GT_dist={s['gt_distance']:.3f}"
            
            axes[5].text(0.1, 0.5, '\n'.join(stats_text), transform=axes[5].transAxes,
                        fontsize=10, verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[5].axis('off')
            axes[5].set_title('Layer Statistics', fontsize=14, fontweight='bold')
            
            plt.suptitle(f'Layer Progression Analysis - Step {self.debug_step}',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_progression.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Layer progression vis error: {e}")

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
        
        # [NEW] 用于逐层追踪的数据收集
        all_layer_outputs_for_vis = []
        all_reference_points_for_vis = []
        all_weights_for_vis = []
        
        # Capture the primary value (usually lidar) from kwargs
        value_lidar = kwargs.get('value', None)
        
        for lid, layer in enumerate(self.layers):
            # [FIX] Ensure output is in correct shape (num_query, bs, embed_dims) at start of each iteration
            # output should be (num_query, bs, embed_dims) for layer input
            if lid > 0:  # After first layer, output might have been permuted
                # Check if output needs to be converted back to (num_query, bs, embed_dims)
                if output.dim() == 3:
                    # If output is (bs, num_query, embed_dims), convert to (num_query, bs, embed_dims)
                    if output.shape[0] != query.shape[0] and output.shape[1] == query.shape[0]:
                        output = output.permute(1, 0, 2)
            
            current_value = value_lidar

            refined_view = None  # 初始化 refined_view
            gate_values = None
            fused_feature_for_vis = None  # 用于可视化的 BEV 融合特征
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
                    
                    # [IMPROVED] Apply temperature scaling to control gate value distribution
                    if self.gate_temperature != 1.0:
                        # Temperature scaling: lower temperature = more conservative (lower gate values)
                        # Formula: sigmoid((x - 0.5) / T + 0.5) where T is temperature
                        gate = torch.sigmoid((gate - 0.5) / self.gate_temperature + 0.5)
                    
                    # [NEW] 更激进的Gating策略：当View特征缺乏空间区分度时，降低gate值
                    # 检测View特征的空间方差（如果方差小，说明特征uniform，应该降低gate）
                    if self.use_adaptive_gating:
                        with torch.no_grad():
                            # 计算每个样本的空间方差（在空间维度H*W上计算方差）
                            view_spatial_var = value_view.var(dim=1).mean()  # 计算空间维度的方差
                            # 如果方差小（<0.1），说明特征uniform（"一片黄色"），应该更保守地使用View
                            if view_spatial_var < 0.1:
                                # 降低gate值，更激进地过滤View特征
                                gate = gate * 0.5  # 将gate值减半
                                # [DEBUG] 只在训练时打印，避免日志过多
                                if self.training and self.debug_step % 1000 == 0:
                                    print(f"[ADAPTIVE GATING] Step {self.debug_step}: View spatial_var={view_spatial_var:.4f} < 0.1, reducing gate by 50%")
                    
                    refined_view = value_view * gate
                    gate_values = gate  # Save for visualization

                # 为了诊断方便，在 BEV 空间上构造一个简单的“融合特征”用于可视化
                if value_lidar is not None:
                    lidar_vis = F.layer_norm(value_lidar, value_lidar.shape[-1:])
                    base_view = refined_view if refined_view is not None else value_view
                    view_vis = F.layer_norm(base_view, base_view.shape[-1:])
                    # 这里使用 0.5/0.5 等权，仅用于可视化，与真实解码器权重解耦
                    fused_feature_for_vis = 0.5 * lidar_vis + 0.5 * view_vis

            # [DEBUG ADDITION]
            if self.training:
                # Pass both Lidar (value_lidar) and View (refined_view OR value_view)
                # We prefer showing refined_view if gating is ON, to see what actually goes into fusion
                vis_view = refined_view if refined_view is not None else value_view
                self.visualize_reference_points(
                    reference_points,
                    lid,
                    feature_map=value_lidar,
                    feature_map_view=vis_view,
                    spatial_shapes=kwargs.get('spatial_shapes', None),
                    input_lidar_img=kwargs.get('input_lidar_img', None),  # [DEBUG VIS] Pass input images
                    input_view_img=kwargs.get('input_view_img', None)  # [DEBUG VIS]
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
                    
                    # [IMPROVED] Feature normalization + Normalized weights for better fusion
                    if value_lidar is not None:
                        # 1. Feature Normalization: 确保两个模态的特征尺度匹配
                        # 使用Layer Normalization，让每个样本的特征分布标准化
                        lidar_norm = F.layer_norm(value_lidar, value_lidar.shape[-1:])
                        view_norm = F.layer_norm(refined_view, refined_view.shape[-1:])
                        
                        # 2. Normalize weights: 确保权重和为1.0，防止特征幅度增大
                        lidar_weight = 1.0
                        total_weight = lidar_weight + view_weight
                        lidar_weight = lidar_weight / total_weight
                        view_weight = view_weight / total_weight
                        
                        # 3. Fused value: 使用归一化后的特征和权重
                        current_value = lidar_weight * lidar_norm + view_weight * view_norm
                    else:
                        current_value = refined_view
                        
                else:
                    # Legacy Hard Split Logic
                    if lid >= self.split_layer_index:
                        current_value = refined_view
            
            # [FIXED] Dual Cross Attention: 正确实现 - 只执行cross attention两次，self-attn和FFN只执行一次
            if self.use_dual_cross_attn and value_view is not None and value_lidar is not None:
                # 确保output形状正确 (num_query, bs, embed_dims)
                if output.dim() == 3:
                    num_query_expected = query.shape[0]
                    if output.shape[0] != num_query_expected and output.shape[1] == num_query_expected:
                        output = output.permute(1, 0, 2)

                reference_points_input = reference_points[..., :2].unsqueeze(2)
                view_mask = kwargs.get('view_mask', None)
                view_key_padding_mask = view_mask if view_mask is not None else key_padding_mask

                # ============ 正确的Dual Cross Attention实现 ============
                # 获取layer内部的attention和ffn模块
                # DetrTransformerDecoderLayer的operation_order: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                # 
                # [IMPORTANT] 实现逻辑：
                # 1. Self-Attention: 只执行一次（query attends to itself）
                # 2. Dual Cross-Attention: 使用同一个cross_attn模块，分别对Lidar和View执行
                #    - 注意：这要求cross_attn是无状态的，可以安全地多次调用
                # 3. 融合两个cross-attention的输出
                # 4. FFN: 只执行一次
                #
                # [FIX] 检查layer是否有必要的属性，以及是否支持pre-norm
                has_attentions = hasattr(layer, 'attentions') and len(layer.attentions) > 0
                has_norms = hasattr(layer, 'norms') and len(layer.norms) > 0
                has_ffns = hasattr(layer, 'ffns') and len(layer.ffns) > 0
                pre_norm = getattr(layer, 'pre_norm', False)  # 检查是否使用pre-norm
                
                # [DEBUG] 验证layer结构
                if self.training and self.debug_step % 1000 == 0 and lid == 0:
                    print(f"[DUAL ATTENTION DEBUG] Layer {lid}: has_attentions={has_attentions}, "
                          f"has_norms={has_norms}, has_ffns={has_ffns}, pre_norm={pre_norm}")
                    if has_attentions:
                        print(f"[DUAL ATTENTION DEBUG] attentions count: {len(layer.attentions)}")
                    if hasattr(layer, 'operation_order'):
                        print(f"[DUAL ATTENTION DEBUG] operation_order: {layer.operation_order}")
                
                if not has_attentions or not has_norms or not has_ffns:
                    # [FALLBACK] 如果无法访问内部属性，回退到标准layer调用
                    # 这种情况下，我们只能对融合后的特征执行一次cross attention
                    kwargs['value'] = current_value
                    output = layer(output, *args, reference_points=reference_points_input,
                                  key_padding_mask=key_padding_mask, **kwargs)
                    output = output.permute(1, 0, 2)  # (bs, num_query, embed_dims)
                else:
                    # Step 1: Self-Attention (只执行一次)
                    identity = output  # 保存用于residual连接
                    self_attn = layer.attentions[0]
                    query_pos = kwargs.get('query_pos', None)

                    # Self-attention: query attends to itself
                    # [FIX] 根据pre_norm决定identity的传递方式
                    output_self = self_attn(
                        query=output,
                        key=output,
                        value=output,
                        identity=identity if pre_norm else None,  # pre-norm: identity在之前传入
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=None,
                        key_padding_mask=None
                    )
                    
                    # [FIX] 根据pre_norm决定residual连接方式
                    if pre_norm:
                        # pre-norm: attention输出直接使用，norm在residual之后
                        output = output_self
                    else:
                        # post-norm: attention输出 + residual，然后norm
                        output = output_self + identity
                    
                    # 应用第一个norm (在self_attn之后)
                    output = layer.norms[0](output)
                    identity = output  # 更新identity用于下一个操作

                    # Step 2: Dual Cross-Attention
                    # [VERIFIED] 使用同一个cross_attn模块，但分别对Lidar和View执行
                    # 验证：CustomMSDeformableAttention是无状态的（forward方法不修改self属性）
                    # - 所有操作都是函数式的
                    # - 没有缓存、计数器等运行时状态
                    # - 唯一潜在问题：Dropout在训练时的随机性（但这通常是有益的正则化）
                    # 结论：可以安全地多次调用同一个cross_attn模块
                    cross_attn = layer.attentions[1]
                    
                    # 2a: Cross-attention to Lidar (主模态)
                    # [FIX] 准备Lidar的kwargs，移除所有显式传递的参数，避免冲突
                    kwargs_lidar = kwargs.copy()
                    kwargs_lidar.pop('key_padding_mask', None)  # 移除，使用显式参数
                    kwargs_lidar.pop('view_mask', None)
                    kwargs_lidar.pop('value', None)  # 移除旧的value，使用新的value_lidar
                    kwargs_lidar.pop('key', None)  # 移除key，避免与显式key=None冲突
                    kwargs_lidar.pop('query', None)  # 移除query，使用显式参数
                    kwargs_lidar.pop('query_pos', None)  # 移除query_pos，使用显式参数
                    kwargs_lidar.pop('identity', None)  # 移除identity，使用显式参数
                    kwargs_lidar.pop('reference_points', None)  # 移除reference_points，使用显式参数
                    kwargs_lidar.pop('spatial_shapes', None)  # 移除spatial_shapes，使用显式参数
                    kwargs_lidar.pop('level_start_index', None)  # 移除level_start_index，使用显式参数
                    
                    # [FIX] 根据pre_norm决定identity传递
                    # CustomMSDeformableAttention在最后会执行: return self.dropout(output) + identity
                    # - 如果identity=None，会使用query作为identity
                    # - pre_norm: 传入identity（self_attn后的output），attention会加identity
                    # - post_norm: 传入None，attention会使用query作为identity（即self_attn后的output）
                    # 所以两种情况下，attention输出都包含了residual连接
                    cross_attn_identity = identity if pre_norm else None
                    
                    output_lidar = cross_attn(
                        query=output,
                        key=None,  # CustomMSDeformableAttention不使用key参数
                        value=value_lidar,  # [FIX] 直接传递value参数，shape应该是 (Len, BS, C)
                        identity=cross_attn_identity,
                        query_pos=kwargs.get('query_pos', None),
                        key_padding_mask=key_padding_mask,
                        reference_points=reference_points_input,
                        spatial_shapes=kwargs.get('spatial_shapes', None),
                        level_start_index=kwargs.get('level_start_index', None),
                        **kwargs_lidar
                    )
                    
                    # 2b: Cross-attention to View (辅助模态)
                    # [FIX] 使用相同的query和cross_attn，但不同的value
                    view_value = refined_view if refined_view is not None else value_view
                    kwargs_view = kwargs.copy()
                    kwargs_view.pop('key_padding_mask', None)  # 移除，使用显式参数
                    kwargs_view.pop('view_mask', None)
                    kwargs_view.pop('value', None)  # 移除旧的value，使用新的view_value
                    kwargs_view.pop('key', None)  # 移除key，避免与显式key=None冲突
                    kwargs_view.pop('query', None)  # 移除query，使用显式参数
                    kwargs_view.pop('query_pos', None)  # 移除query_pos，使用显式参数
                    kwargs_view.pop('identity', None)  # 移除identity，使用显式参数
                    kwargs_view.pop('reference_points', None)  # 移除reference_points，使用显式参数
                    kwargs_view.pop('spatial_shapes', None)  # 移除spatial_shapes，使用显式参数
                    kwargs_view.pop('level_start_index', None)  # 移除level_start_index，使用显式参数
                    
                    output_view = cross_attn(
                        query=output,
                        key=None,  # CustomMSDeformableAttention不使用key参数
                        value=view_value,  # [FIX] 直接传递value参数，shape应该是 (Len, BS, C)
                        identity=cross_attn_identity,  # 使用相同的identity
                        query_pos=kwargs.get('query_pos', None),
                        key_padding_mask=view_key_padding_mask,
                        reference_points=reference_points_input,
                        spatial_shapes=kwargs.get('spatial_shapes', None),
                        level_start_index=kwargs.get('level_start_index', None),
                        **kwargs_view
                    )

                    # Step 3: 融合两个cross-attention的输出
                    # 转换为 (bs, num_query, embed_dims) 进行融合
                    output_lidar_t = output_lidar.permute(1, 0, 2)
                    output_view_t = output_view.permute(1, 0, 2)

                    # 计算融合权重
                    if self.use_adaptive_weights:
                        # [FIX] Layer Norm仅用于计算权重和尺度匹配，但融合时使用原始特征
                        # 避免双重归一化：融合前归一化用于权重计算，融合后由layer.norms[1]统一归一化
                        # [FIX] 关于residual连接和梯度流的说明：
                        # - output_lidar_t和output_view_t已经包含了residual（来自CustomMSDeformableAttention）
                        # - 归一化用于权重计算，虽然会影响梯度流，但这是有益的（归一化本身就是为了稳定训练）
                        # - 融合时使用原始特征（output_lidar_t, output_view_t），保持residual连接的完整性
                        # - 融合后由layer.norms[1]统一归一化，这是标准的Transformer做法
                        # 注意：归一化操作是可微的，梯度可以正常反向传播，不会"破坏"residual连接
                        output_lidar_norm = F.layer_norm(output_lidar_t, output_lidar_t.shape[-1:])
                        output_view_norm = F.layer_norm(output_view_t, output_view_t.shape[-1:])
                        concat_feat = torch.cat([output_lidar_norm, output_view_norm], dim=-1)
                        weights = self.adaptive_weight_module(concat_feat)
                        lidar_weight = weights[..., 0:1]
                        view_weight = weights[..., 1:2]
                        
                        # [NEW] 可视化自适应融合权重
                        if self.training:
                            self.visualize_adaptive_weights(
                                lidar_weight, view_weight, reference_points, lid,
                                gt_bboxes_3d=kwargs.get('gt_bboxes_3d', None),
                                spatial_shapes=kwargs.get('spatial_shapes', None)
                            )
                        
                        # [FIX] 使用原始特征进行融合，避免双重归一化
                        # 归一化仅用于权重计算，融合后由layer.norms[1]统一归一化
                        fused_cross = lidar_weight * output_lidar_t + view_weight * output_view_t
                        
                        # [NEW] 可视化融合前后特征对比（使用归一化后的特征用于可视化）
                        if self.training:
                            self.visualize_feature_comparison(
                                output_lidar_norm, output_view_norm, 
                                F.layer_norm(fused_cross, fused_cross.shape[-1:]),  # 临时归一化用于可视化
                                reference_points, lid,
                                gt_bboxes_3d=kwargs.get('gt_bboxes_3d', None),
                                spatial_shapes=kwargs.get('spatial_shapes', None)
                            )
                    else:
                        # 渐进式融合或固定权重
                        num_layers = len(self.layers)
                        if self.use_gradual:
                            view_weight = float(lid) / float(num_layers - 1) if num_layers > 1 else 0.5
                            lidar_weight = 1.0 - view_weight
                        else:
                            lidar_weight, view_weight = 0.7, 0.3  # Lidar主导

                        # [FIX] 直接使用原始特征进行融合，避免双重归一化
                        # 融合后由layer.norms[1]统一归一化（标准Transformer做法）
                        fused_cross = lidar_weight * output_lidar_t + view_weight * output_view_t
                        
                        # [NEW] 对于非adaptive weights情况，也收集权重用于可视化
                        if self.training:
                            # 创建权重tensor用于可视化
                            lidar_weight_tensor = torch.full((output_lidar_t.shape[0], output_lidar_t.shape[1], 1), 
                                                             lidar_weight, device=output_lidar_t.device)
                            view_weight_tensor = torch.full((output_view_t.shape[0], output_view_t.shape[1], 1),
                                                            view_weight, device=output_view_t.device)
                            
                            # 可视化融合前后特征对比（临时归一化用于可视化）
                            output_lidar_norm_vis = F.layer_norm(output_lidar_t, output_lidar_t.shape[-1:])
                            output_view_norm_vis = F.layer_norm(output_view_t, output_view_t.shape[-1:])
                            fused_cross_norm_vis = F.layer_norm(fused_cross, fused_cross.shape[-1:])
                            self.visualize_feature_comparison(
                                output_lidar_norm_vis, output_view_norm_vis, fused_cross_norm_vis, 
                                reference_points, lid,
                                gt_bboxes_3d=kwargs.get('gt_bboxes_3d', None),
                                spatial_shapes=kwargs.get('spatial_shapes', None)
                            )

                    # 转回 (num_query, bs, embed_dims)
                    fused_cross = fused_cross.permute(1, 0, 2)

                    # [FIX] 应用cross-attention后的norm (第二个norm)
                    # 注意：CustomMSDeformableAttention已经在最后执行了: return self.dropout(output) + identity
                    # - pre_norm: output_lidar = attention_output + identity（identity是self_attn后的output）
                    # - post_norm: output_lidar = attention_output + query（query是self_attn后的output）
                    # 所以output_lidar和output_view都已经包含了residual连接
                    # 融合后：fused_cross = lidar_weight * output_lidar + view_weight * output_view
                    # 这个fused_cross已经包含了residual，现在统一归一化（避免双重归一化）
                    # 融合前如果做了归一化，仅用于权重计算，融合使用原始特征
                    output = layer.norms[1](fused_cross)
                    identity = output  # 更新identity用于FFN

                # Step 4: FFN (只执行一次)
                if has_ffns:
                    # [FIX] FFN的调用方式：FFN可能接受identity参数（pre_norm）或需要手动加residual（post_norm）
                    # 检查FFN的forward签名
                    ffn_identity = identity if pre_norm else None
                    try:
                        # 尝试传递identity参数（如果FFN支持pre_norm）
                        ffn_output = layer.ffns[0](output, ffn_identity)
                    except TypeError:
                        # 如果FFN不支持identity参数，手动处理
                        ffn_output = layer.ffns[0](output)
                        if not pre_norm:
                            ffn_output = ffn_output + identity
                    
                    # [FIX] 应用FFN后的norm (第三个norm)
                    # pre_norm: ffn_output已经包含了residual（如果FFN支持），直接norm
                    # post_norm: ffn_output已经手动加了residual，直接norm
                    output = layer.norms[2](ffn_output)

                # 转换为 (bs, num_query, embed_dims) 用于reg_branches
                output = output.permute(1, 0, 2)
                
                # [NEW] 收集数据用于逐层追踪可视化
                if self.training:
                    # 保存当前层的输出、reference points和权重
                    all_layer_outputs_for_vis.append(output.clone().detach())
                    all_reference_points_for_vis.append(reference_points.clone().detach())
                    # 保存权重（如果存在）
                    if 'lidar_weight' in locals() and 'view_weight' in locals():
                        if isinstance(lidar_weight, torch.Tensor):
                            all_weights_for_vis.append((lidar_weight.clone().detach(), view_weight.clone().detach()))
                        else:
                            # 对于float类型的权重，创建tensor
                            lidar_w_tensor = torch.tensor(lidar_weight, device=output.device)
                            view_w_tensor = torch.tensor(view_weight, device=output.device)
                            all_weights_for_vis.append((lidar_w_tensor, view_w_tensor))
                    else:
                        all_weights_for_vis.append(None)

            else:
                # Original single cross attention (fused features)
                # Update kwargs locally for this layer call
                kwargs['value'] = current_value

                reference_points_input = reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
                
                output = layer(
                    output,
                    *args,
                    reference_points=reference_points_input,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                output = output.permute(1, 0, 2)  # (bs, num_query, embed_dims)

            if reg_branches is not None:
                # [FIX] output should be (bs, num_query, embed_dims) for reg_branches
                # In dual cross attention branch, output is already (bs, num_query, embed_dims)
                # In single cross attention branch, output is (bs, num_query, embed_dims) after permute
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            # [FIX] Convert output to (num_query, bs, embed_dims) for next layer
            # Both branches have output as (bs, num_query, embed_dims) at this point
            output = output.permute(1, 0, 2)  # (num_query, bs, embed_dims)

            # [DEBUG ADDITION]
            if self.training:
                # Pass both Lidar (value_lidar) and View (refined_view OR value_view)
                # We prefer showing refined_view if gating is ON, to see what actually goes into fusion
                vis_view = refined_view if refined_view is not None else value_view
                
                # Get view mask from kwargs if available
                view_mask_vis = kwargs.get('view_mask', None)
                
                self.visualize_reference_points(
                    reference_points, 
                    lid, 
                    feature_map=value_lidar, 
                    feature_map_view=vis_view,
                    spatial_shapes=kwargs.get('spatial_shapes', None),
                    gt_bboxes_3d=kwargs.get('gt_bboxes_3d', None),
                    gate_values=gate_values,
                    fused_feature=fused_feature_for_vis,
                    view_mask=view_mask_vis,
                    input_lidar_img=kwargs.get('input_lidar_img', None),
                    input_view_img=kwargs.get('input_view_img', None)
                )

            # [FIX] output should be (num_query, bs, embed_dims) when appending to intermediate
            # This matches the original decoder behavior
            if self.return_intermediate:
                intermediate.append(output)  # output is already (num_query, bs, embed_dims)
                intermediate_reference_points.append(reference_points)

        # [NEW] 可视化逐层融合效果追踪
        if self.training and len(all_layer_outputs_for_vis) > 0:
            self.visualize_layer_progression(
                all_layer_outputs_for_vis, all_reference_points_for_vis, all_weights_for_vis,
                gt_bboxes_3d=kwargs.get('gt_bboxes_3d', None)
            )
        
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
                gt_bboxes_3d=None, # [DEBUG VIS]
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
        mask_flatten_view = None
        if mlvl_feats_view is not None:
            feat_flatten_view = []
            mask_flatten_view_list = []
            
            # [FIX] Ensure View features are spatially aligned with Lidar features
            # Critical: View features must have the same spatial dimensions as Lidar
            import torch.nn.functional as F
            
            # [FIX] Generate View mask similar to Lidar mask (based on img_shape from img_metas)
            # Reference: vmahead.py 
            mlvl_masks_view = None
            if kwargs.get('img_metas', None) is not None:
                img_metas = kwargs.get('img_metas')
                input_img_h, input_img_w = img_metas[0]['img_shape']
                # Create mask: 1=invalid (padding), 0=valid
                # Same logic as lidar mask generation
                img_masks_view = mlvl_feats_view[0].new_ones((bs, input_img_h, input_img_w))
                for img_id in range(bs):
                    img_h, img_w = img_metas[img_id]['img_shape']
                    img_masks_view[img_id, :img_h, :img_w] = 0
                
                mlvl_masks_view = []
                for lvl, feat_view in enumerate(mlvl_feats_view):
                    # [FIX] First align View feature size to Lidar feature size
                    # NOTE: View features should already be aligned in vmahead.py
                    # This is a safety check for mask generation
                    feat_lidar = mlvl_feats[lvl]  # Get corresponding Lidar feature
                    lidar_h, lidar_w = feat_lidar.shape[-2:]
                    view_h, view_w = feat_view.shape[-2:]
                    
                    # Interpolate View feature to match Lidar spatial dimensions (safety fallback)
                    if (view_h, view_w) != (lidar_h, lidar_w):
                        # This should rarely happen if vmahead.py alignment works correctly
                        feat_view = F.interpolate(
                            feat_view, 
                            size=(lidar_h, lidar_w), 
                            mode='bilinear',  # Match vmahead.py interpolation mode
                            align_corners=False  # Match vmahead.py align_corners setting
                        )
                    
                    # Interpolate mask to match feature map size (after alignment)
                    mask_view = F.interpolate(
                        img_masks_view.unsqueeze(1).float(), 
                        size=(lidar_h, lidar_w)
                    ).to(torch.bool).squeeze(1)
                    mlvl_masks_view.append(mask_view)
            
            # [FIX] Process View features with spatial alignment check
            # NOTE: View features should already be aligned in vmahead.py (interpolate + high_res_smoothing)
            # This is a safety check to ensure alignment, but should rarely trigger if vmahead.py works correctly
            for lvl, feat in enumerate(mlvl_feats_view):
                # [FIX] Ensure View feature matches Lidar spatial dimensions
                feat_lidar = mlvl_feats[lvl]
                lidar_h, lidar_w = feat_lidar.shape[-2:]
                view_h, view_w = feat.shape[-2:]
                
                # [DEBUG] Log spatial dimension mismatch (should be rare if vmahead.py alignment works)
                if (view_h, view_w) != (lidar_h, lidar_w):
                    if self.training and lvl == 0:  # Only log for first level to avoid spam
                        print(f"[WARNING] Level {lvl}: View ({view_h}, {view_w}) != Lidar ({lidar_h}, {lidar_w})")
                        print(f"[WARNING] This suggests vmahead.py alignment may have failed. Re-aligning in split_modules...")
                    # [FIX] Align View feature to Lidar dimensions (safety fallback)
                    # Use same interpolation method as vmahead.py for consistency
                    feat = F.interpolate(
                        feat, 
                        size=(lidar_h, lidar_w), 
                        mode='bilinear',  # Match vmahead.py interpolation mode
                        align_corners=False  # Match vmahead.py align_corners setting
                    )
                
                feat = feat.flatten(2).transpose(1, 2)  # (BS, H*W, C)
                
                # [FIX] Use View mask similar to Lidar mask (based on img_shape)
                # If mlvl_masks_view is available, use it; otherwise fallback to lidar mask or feature-based mask
                if mlvl_masks_view is not None and lvl < len(mlvl_masks_view):
                    # Use view mask generated from img_shape (same logic as lidar)
                    view_mask = mlvl_masks_view[lvl].flatten(1)  # (BS, H*W) bool, True=invalid
                elif lvl < len(mlvl_masks):
                    # Fallback: use lidar mask if view mask not available
                    view_mask = mlvl_masks[lvl].flatten(1)  # (BS, H*W) bool, True=invalid
                else:
                    # Last resort: generate mask based on feature magnitude
                    with torch.no_grad():
                        feat_magnitude = torch.norm(feat, p=2, dim=-1)  # (BS, H*W)
                        view_valid = feat_magnitude > 1e-4  # (BS, H*W) bool
                        view_mask = ~view_valid  # (BS, H*W) bool, True=invalid
                
                mask_flatten_view_list.append(view_mask)
                feat_flatten_view.append(feat)
            
            # Combine levels
            feat_flatten_view = torch.cat(feat_flatten_view, 1) # (BS, Total_Len, C)
            mask_flatten_view = torch.cat(mask_flatten_view_list, 1)  # (BS, Total_Len)
            
            # Pass through View Encoder if it exists
            if self.encoder_view is not None:
                # [FIX] Use View's own mask instead of Lidar mask
                # [FIX] Ensure View uses the same spatial_shapes as Lidar (after alignment)
                # After interpolation, View features should have the same spatial dimensions as Lidar
                memory_view = self.encoder_view(
                    query=feat_flatten_view.permute(1, 0, 2), # (Len, BS, C)
                    key=None,
                    value=None,
                    query_pos=lvl_pos_embed_flatten.permute(1, 0, 2), # [FIX] Reuse pos embeds (same spatial structure after alignment)
                    query_key_padding_mask=mask_flatten_view,          # [FIX] Use View mask
                    spatial_shapes=spatial_shapes,  # [FIX] Use same spatial_shapes (after alignment, they match)
                    level_start_index=level_start_index,  # [FIX] Same level_start_index (same spatial structure)
                    valid_ratios=valid_ratios,  # [FIX] Could use View's own valid_ratios, but using Lidar's for consistency
                    reference_points=reference_points,  # [FIX] Same reference points (same BEV space)
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

        # [DEBUG VIS] Extract input images from kwargs to avoid duplicate arguments
        input_lidar_img = kwargs.pop('input_lidar_img', None)
        input_view_img = kwargs.pop('input_view_img', None)

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
            gt_bboxes_3d=gt_bboxes_3d, # [DEBUG VIS]
            view_mask=mask_flatten_view,  # [FIX] Pass view mask to decoder
            input_lidar_img=input_lidar_img,  # [DEBUG VIS] Pass input images
            input_view_img=input_view_img,  # [DEBUG VIS]
            **kwargs)
            
        return inter_states, init_reference_out, inter_references, None, None
