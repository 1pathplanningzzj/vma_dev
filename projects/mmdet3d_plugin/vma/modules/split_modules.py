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
        
        # [FIX] Get embed_dims from kwargs if provided, otherwise use parameter value
        if 'embed_dims' in kwargs:
            embed_dims = kwargs.pop('embed_dims')  # Use kwargs value and remove it
        # Otherwise use the parameter value (already set as embed_dims=256)
        
        # [NEW] Adaptive weight learning module
        if self.use_adaptive_weights:
            self.adaptive_weight_module = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 2),  # Output: [lidar_weight, view_weight]
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
        self.debug_dir = "debug_vis_gt_mask_3/attention_check"
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
            
            # [NEW] Dual Cross Attention: Query attends to Lidar (main) and View (auxiliary) separately
            if self.use_dual_cross_attn and value_view is not None and value_lidar is not None:
                # [FIX] Ensure output is in correct shape (num_query, bs, embed_dims) for layer
                # output should be (num_query, bs, embed_dims) at the start of each layer iteration
                # Check and fix output shape if needed
                if output.dim() == 3:
                    num_query_expected = query.shape[0]  # Expected num_query dimension
                    if output.shape[0] != num_query_expected:
                        # If output is (bs, num_query, embed_dims), convert to (num_query, bs, embed_dims)
                        if output.shape[1] == num_query_expected:
                            output = output.permute(1, 0, 2)
                        else:
                            # Unexpected shape, try to infer
                            raise RuntimeError(f"Unexpected output shape: {output.shape}, expected first dim to be {num_query_expected}")
                
                # Save original output for residual connection
                output_orig = output
                reference_points_input = reference_points[..., :2].unsqueeze(2)
                view_mask = kwargs.get('view_mask', None)
                
                # [IMPROVED] Process Lidar first (main modality), then View (auxiliary)
                if self.lidar_first:
                    # Step 1: Query attends to Lidar features (main modality)
                    kwargs_lidar = kwargs.copy()
                    kwargs_lidar['value'] = value_lidar
                    # [FIX] Remove key_padding_mask from kwargs to avoid duplicate argument
                    kwargs_lidar.pop('key_padding_mask', None)
                    
                    output_lidar = layer(
                        output,  # (num_query, bs, embed_dims)
                        *args,
                        reference_points=reference_points_input,
                        key_padding_mask=key_padding_mask,  # Pass explicitly
                        **kwargs_lidar)
                    output_lidar = output_lidar.permute(1, 0, 2)  # (bs, num_query, embed_dims)
                    
                    # [NEW] Modality Interaction: Use Lidar output to guide View attention
                    if self.use_modality_interaction:
                        # Extract query features from Lidar output for interaction
                        query_for_view = output_lidar  # (bs, num_query, embed_dims)
                    else:
                        query_for_view = output.permute(1, 0, 2)  # Original query
                    
                    # Step 2: Query attends to View features (auxiliary modality)
                    kwargs_view = kwargs.copy()
                    kwargs_view['value'] = refined_view if refined_view is not None else value_view
                    # [FIX] Remove key_padding_mask from kwargs to avoid duplicate argument
                    kwargs_view.pop('key_padding_mask', None)
                    # Determine which mask to use
                    view_key_padding_mask = view_mask if view_mask is not None else key_padding_mask
                    
                    # Use query_for_view (may be enhanced by Lidar output)
                    output_view = layer(
                        query_for_view.permute(1, 0, 2),  # Convert to (num_query, bs, embed_dims)
                        *args,
                        reference_points=reference_points_input,
                        key_padding_mask=view_key_padding_mask,  # Pass explicitly
                        **kwargs_view)
                    output_view = output_view.permute(1, 0, 2)  # (bs, num_query, embed_dims)
                else:
                    # Original order: View first, then Lidar
                    kwargs_view = kwargs.copy()
                    kwargs_view['value'] = refined_view if refined_view is not None else value_view
                    # [FIX] Remove key_padding_mask from kwargs to avoid duplicate argument
                    kwargs_view.pop('key_padding_mask', None)
                    view_key_padding_mask = view_mask if view_mask is not None else key_padding_mask
                    
                    output_view = layer(
                        output,
                        *args,
                        reference_points=reference_points_input,
                        key_padding_mask=view_key_padding_mask,  # Pass explicitly
                        **kwargs_view)
                    output_view = output_view.permute(1, 0, 2)
                    
                    kwargs_lidar = kwargs.copy()
                    kwargs_lidar['value'] = value_lidar
                    # [FIX] Remove key_padding_mask from kwargs to avoid duplicate argument
                    kwargs_lidar.pop('key_padding_mask', None)
                    
                    output_lidar = layer(
                        output_view.permute(1, 0, 2),
                        *args,
                        reference_points=reference_points_input,
                        key_padding_mask=key_padding_mask,  # Pass explicitly
                        **kwargs_lidar)
                    output_lidar = output_lidar.permute(1, 0, 2)
                
                # Step 3: Fuse outputs from Lidar and View attention
                # Normalize features before fusion
                output_lidar_norm = F.layer_norm(output_lidar, output_lidar.shape[-1:])
                output_view_norm = F.layer_norm(output_view, output_view.shape[-1:])
                
                # [NEW] Adaptive Weight Learning: Learn weights based on features
                if self.use_adaptive_weights:
                    # Concatenate normalized features to predict weights
                    concat_feat = torch.cat([output_lidar_norm, output_view_norm], dim=-1)  # (bs, num_query, 2*embed_dims)
                    # Average over query dimension to get global feature
                    global_feat = concat_feat.mean(dim=1)  # (bs, 2*embed_dims)
                    weights = self.adaptive_weight_module(global_feat)  # (bs, 2) -> [lidar_weight, view_weight]
                    # Expand to match query dimension
                    weights = weights.unsqueeze(1)  # (bs, 1, 2)
                    lidar_weight = weights[..., 0:1]  # (bs, 1, 1)
                    view_weight = weights[..., 1:2]  # (bs, 1, 1)
                elif self.use_gradual:
                    # Gradual fusion weights (layer-dependent)
                    num_layers = len(self.layers)
                    view_weight_val = float(lid) / float(num_layers - 1) if num_layers > 1 else 1.0
                    view_weight_val = min(max(view_weight_val, 0.0), 1.0)
                    lidar_weight_val = 1.0 - view_weight_val
                    lidar_weight = lidar_weight_val
                    view_weight = view_weight_val
                else:
                    # Fixed weights (favor Lidar as main modality)
                    lidar_weight = 0.6
                    view_weight = 0.4
                
                # [NEW] Conditional Fusion: Query-dependent modality gating
                if self.use_conditional_fusion:
                    # Use fused output (before final permute) to predict modality gates
                    # output_lidar_norm and output_view_norm are both (bs, num_query, embed_dims)
                    # Use their average or concatenation to get query representation
                    query_flat = (output_lidar_norm + output_view_norm) / 2.0  # (bs, num_query, embed_dims)
                    query_global = query_flat.mean(dim=1)  # (bs, embed_dims)
                    modality_gates = self.conditional_fusion(query_global)  # (bs, 2) -> [lidar_gate, view_gate]
                    lidar_gate = modality_gates[:, 0:1].unsqueeze(1)  # (bs, 1, 1)
                    view_gate = modality_gates[:, 1:2].unsqueeze(1)  # (bs, 1, 1)
                    
                    # Apply gates to weights
                    if isinstance(lidar_weight, float):
                        lidar_weight = lidar_weight * lidar_gate
                    else:
                        lidar_weight = lidar_weight * lidar_gate
                    if isinstance(view_weight, float):
                        view_weight = view_weight * view_gate
                    else:
                        view_weight = view_weight * view_gate
                    
                    # Renormalize
                    total_weight = lidar_weight + view_weight
                    lidar_weight = lidar_weight / (total_weight + 1e-8)
                    view_weight = view_weight / (total_weight + 1e-8)
                
                # Weighted fusion
                if isinstance(lidar_weight, float):
                    output = lidar_weight * output_lidar_norm + view_weight * output_view_norm
                else:
                    output = lidar_weight * output_lidar_norm + view_weight * output_view_norm
                
                # [NEW] Modality Interaction: Cross-modal enhancement
                if self.use_modality_interaction and not self.lidar_first:
                    # Enhance fused output with cross-modal interaction
                    concat_output = torch.cat([output_lidar_norm, output_view_norm], dim=-1)
                    interaction_feat = self.modality_interaction(concat_output)
                    output = output + 0.1 * interaction_feat  # Residual connection with small weight
                
                # [FIX] output is currently (bs, num_query, embed_dims)
                # Keep it as (bs, num_query, embed_dims) for reg_branches
                # Then convert to (num_query, bs, embed_dims) for next layer
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
            
            # [FIX] Generate View mask similar to Lidar mask (based on img_shape from img_metas)
            # Reference: vmahead.py 
            mlvl_masks_view = None
            if kwargs.get('img_metas', None) is not None:
                img_metas = kwargs.get('img_metas')
                import torch.nn.functional as F
                input_img_h, input_img_w = img_metas[0]['img_shape']
                # Create mask: 1=invalid (padding), 0=valid
                # Same logic as lidar mask generation
                img_masks_view = mlvl_feats_view[0].new_ones((bs, input_img_h, input_img_w))
                for img_id in range(bs):
                    img_h, img_w = img_metas[img_id]['img_shape']
                    img_masks_view[img_id, :img_h, :img_w] = 0
                
                mlvl_masks_view = []
                for feat_view in mlvl_feats_view:
                    # Interpolate mask to match feature map size
                    mask_view = F.interpolate(img_masks_view[None], size=feat_view.shape[-2:]).to(torch.bool).squeeze(0)
                    mlvl_masks_view.append(mask_view)
            
            for lvl, feat in enumerate(mlvl_feats_view):
                # Assume alignment with Lidar
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
                memory_view = self.encoder_view(
                    query=feat_flatten_view.permute(1, 0, 2), # (Len, BS, C)
                    key=None,
                    value=None,
                    query_pos=lvl_pos_embed_flatten.permute(1, 0, 2), # Reuse pos embeds
                    query_key_padding_mask=mask_flatten_view,          # [FIX] Use View mask
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
