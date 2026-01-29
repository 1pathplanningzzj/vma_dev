import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2  # 用于热力图平滑和缩放
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.transformer import DeformableDetrTransformer, inverse_sigmoid
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from .decoder import VMADetectionTransformerDecoder
# date 2026-01-15
# author: zijian
# email: zhangzijian@trunk.tech
# description: This file is the implementation of the SplitModalityDecoder class.
# It is used to decode the features of the lidar and view modalities.
# It is also used to fuse the features of the lidar and view modalities.
# It is also used to predict the bounding boxes of the objects.
# It is also used to predict the classes of the objects.
# It is also used to predict the scores of the objects.
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
            
            # [NEW] Additional visualizations
            # 1. Feature comparison visualization
            if feature_map is not None and feature_map_view is not None:
                self.visualize_feature_comparison(
                    reference_points, layer_idx,
                    feature_map, feature_map_view, fused_feature,
                    spatial_shapes, gt_bboxes_3d
                )
            
            # 2. Adaptive weights visualization (if using adaptive weights)
            if self.use_adaptive_weights and hasattr(self, '_last_adaptive_weights'):
                self.visualize_adaptive_weights(
                    reference_points, layer_idx,
                    self._last_adaptive_weights, spatial_shapes
                )
            
            # 3. Cross-attention visualization
            if feature_map is not None or feature_map_view is not None:
                # Get query from current output (approximate)
                query_for_vis = None
                if hasattr(self, '_last_query'):
                    query_for_vis = self._last_query
                self.visualize_cross_attention(
                    reference_points, layer_idx,
                    query_for_vis, feature_map, feature_map_view, spatial_shapes
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
            
            num_plots = 2
            if gate_values is not None:
                num_plots += 1
            if view_mask is not None:
                num_plots += 1
            if fused_feature is not None:
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
            # [FIX] Calculate statistics based on feature vector L2 norms, not raw feature values
            stats_text = []
            if feature_map_lidar is not None:
                # Compute L2 norm for each feature vector: (BS, Num_Keys, C) -> (BS, Num_Keys)
                lidar_norms = torch.norm(feature_map_lidar, p=2, dim=-1)  # (BS, Num_Keys)
                lidar_mean = lidar_norms.mean().item()
                lidar_std = lidar_norms.std(unbiased=False).item()  # Use unbiased=False for consistency
                stats_text.append(f"Lidar: μ={lidar_mean:.3f}, σ={lidar_std:.3f}")
            if feature_map_view is not None:
                view_norms = torch.norm(feature_map_view, p=2, dim=-1)  # (BS, Num_Keys)
                view_mean = view_norms.mean().item()
                view_std = view_norms.std(unbiased=False).item()
                stats_text.append(f"View: μ={view_mean:.3f}, σ={view_std:.3f}")
            if fused_feature is not None:
                fused_norms = torch.norm(fused_feature, p=2, dim=-1)  # (BS, Num_Keys)
                fused_mean = fused_norms.mean().item()
                fused_std = fused_norms.std(unbiased=False).item()
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

    def visualize_feature_comparison(self, reference_points, layer_idx,
                                     feature_map_lidar, feature_map_view, fused_feature,
                                     spatial_shapes, gt_bboxes_3d):
        """Visualize feature comparison: magnitude distribution, alignment, fusion effect"""
        if self.debug_step % 1000 != 0:
            return
        
        try:
            # Compute feature norms
            if feature_map_lidar is not None:
                lidar_norms = torch.norm(feature_map_lidar, p=2, dim=-1).detach().cpu().numpy()  # (BS, Num_Keys)
                lidar_norms_flat = lidar_norms.flatten()
            else:
                lidar_norms_flat = None
                
            if feature_map_view is not None:
                view_norms = torch.norm(feature_map_view, p=2, dim=-1).detach().cpu().numpy()
                view_norms_flat = view_norms.flatten()
            else:
                view_norms_flat = None
                
            if fused_feature is not None:
                fused_norms = torch.norm(fused_feature, p=2, dim=-1).detach().cpu().numpy()
                fused_norms_flat = fused_norms.flatten()
            else:
                fused_norms_flat = None
            
            # Compute GT distances if available
            gt_distances = None
            if gt_bboxes_3d is not None and reference_points is not None:
                try:
                    pts = reference_points[0].detach().cpu().numpy()  # (Num_Query, 2)
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
                            # Compute distance from each query point to nearest GT point
                            try:
                                from scipy.spatial.distance import cdist
                                distances = cdist(pts, gt_pts)
                                gt_distances = distances.min(axis=1)  # (Num_Query,)
                            except ImportError:
                                # Fallback: simple euclidean distance
                                distances = np.sqrt(((pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=-1))
                                gt_distances = distances.min(axis=1)
                except:
                    try:
                        # Fallback: simple euclidean distance
                        if hasattr(gt, 'tensor'):
                            gt_pts = gt.tensor[0, :, :2].cpu().numpy()
                        else:
                            gt_pts = gt[0, :, :2].cpu().numpy()
                        distances = np.sqrt(((pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=-1))
                        gt_distances = distances.min(axis=1)
                    except:
                        pass
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Plot 1: Feature Magnitude Distribution
            ax = axes[0, 0]
            if lidar_norms_flat is not None:
                ax.hist(lidar_norms_flat, bins=50, alpha=0.6, label='Lidar', color='green', density=True)
            if view_norms_flat is not None:
                ax.hist(view_norms_flat, bins=50, alpha=0.6, label='View', color='red', density=True)
            if fused_norms_flat is not None:
                ax.hist(fused_norms_flat, bins=50, alpha=0.6, label='Fused', color='blue', density=True)
            ax.set_xlabel('Feature Magnitude (L2 Norm)')
            ax.set_ylabel('Density')
            ax.set_title('Feature Magnitude Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Lidar vs View Feature Magnitude
            ax = axes[0, 1]
            if lidar_norms_flat is not None and view_norms_flat is not None:
                # Sample for visualization if too many points
                if len(lidar_norms_flat) > 10000:
                    indices = np.random.choice(len(lidar_norms_flat), 10000, replace=False)
                    ax.scatter(lidar_norms_flat[indices], view_norms_flat[indices], 
                             alpha=0.3, s=1, c='blue')
                else:
                    ax.scatter(lidar_norms_flat, view_norms_flat, alpha=0.3, s=1, c='blue')
                # Add y=x line
                max_val = max(lidar_norms_flat.max(), view_norms_flat.max())
                ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
            ax.set_xlabel('Lidar Feature Magnitude')
            ax.set_ylabel('View Feature Magnitude')
            ax.set_title('Lidar vs View Feature Magnitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Fusion Effect: Input vs Output
            ax = axes[0, 2]
            if fused_norms_flat is not None:
                if lidar_norms_flat is not None:
                    if len(lidar_norms_flat) > 10000:
                        indices = np.random.choice(len(lidar_norms_flat), 10000, replace=False)
                        ax.scatter(lidar_norms_flat[indices], fused_norms_flat[indices], 
                                 alpha=0.3, s=1, c='green', label='Lidar->Fused')
                    else:
                        ax.scatter(lidar_norms_flat, fused_norms_flat, 
                                 alpha=0.3, s=1, c='green', label='Lidar->Fused')
                if view_norms_flat is not None:
                    if len(view_norms_flat) > 10000:
                        indices = np.random.choice(len(view_norms_flat), 10000, replace=False)
                        ax.scatter(view_norms_flat[indices], fused_norms_flat[indices], 
                                 alpha=0.3, s=1, c='red', label='View->Fused')
                    else:
                        ax.scatter(view_norms_flat, fused_norms_flat, 
                                 alpha=0.3, s=1, c='red', label='View->Fused')
                max_val = fused_norms_flat.max()
                ax.plot([0, max_val], [0, max_val], 'b--', linewidth=2, label='y=x')
            ax.set_xlabel('Input Feature Magnitude')
            ax.set_ylabel('Fused Feature Magnitude')
            ax.set_title('Fusion Effect: Input vs Output')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Feature Quality vs GT Distance
            ax = axes[1, 0]
            if gt_distances is not None:
                # Normalize distances
                gt_distances_norm = gt_distances / (gt_distances.max() + 1e-9)
                if lidar_norms_flat is not None and len(lidar_norms_flat) == len(gt_distances_norm):
                    ax.scatter(gt_distances_norm, lidar_norms_flat, alpha=0.3, s=1, c='green', label='Lidar')
                if view_norms_flat is not None and len(view_norms_flat) == len(gt_distances_norm):
                    ax.scatter(gt_distances_norm, view_norms_flat, alpha=0.3, s=1, c='red', label='View')
                if fused_norms_flat is not None and len(fused_norms_flat) == len(gt_distances_norm):
                    ax.scatter(gt_distances_norm, fused_norms_flat, alpha=0.3, s=1, c='blue', label='Fused')
            ax.set_xlabel('Normalized GT Distance')
            ax.set_ylabel('Feature Magnitude')
            ax.set_title('Feature Quality vs GT Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 5: Feature Statistics
            ax = axes[1, 1]
            stats_text = []
            if lidar_norms_flat is not None:
                lidar_mean = lidar_norms_flat.mean()
                lidar_std = lidar_norms_flat.std()
                stats_text.append(f"Lidar: μ={lidar_mean:.3f}, σ={lidar_std:.3f}")
            if view_norms_flat is not None:
                view_mean = view_norms_flat.mean()
                view_std = view_norms_flat.std()
                stats_text.append(f"View: μ={view_mean:.3f}, σ={view_std:.3f}")
            if fused_norms_flat is not None:
                fused_mean = fused_norms_flat.mean()
                fused_std = fused_norms_flat.std()
                stats_text.append(f"Fused: μ={fused_mean:.3f}, σ={fused_std:.3f}")
                if lidar_norms_flat is not None and view_norms_flat is not None:
                    fusion_gain = fused_mean / ((lidar_mean + view_mean) / 2 + 1e-9)
                    stats_text.append(f"Fusion Gain: {fusion_gain:.3f}")
            
            ax.text(0.1, 0.5, '\n'.join(stats_text), transform=ax.transAxes,
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.axis('off')
            ax.set_title("Feature Statistics")
            
            # Plot 6: Fused Feature Spatial Distribution
            ax = axes[1, 2]
            if fused_feature is not None and spatial_shapes is not None:
                try:
                    # Process fused feature for visualization
                    if fused_feature.dim() == 3:
                        if fused_feature.shape[1] != fused_feature.shape[0] and fused_feature.shape[0] > fused_feature.shape[1]:
                            seq_dim = 0
                        else:
                            seq_dim = 1
                    else:
                        seq_dim = 1
                    
                    current_len = fused_feature.shape[seq_dim]
                    H, W = spatial_shapes[0].tolist()
                    feat_len = H * W
                    
                    if current_len >= feat_len:
                        if seq_dim == 0:
                            feat = fused_feature[:feat_len, 0, :].detach().cpu()
                        else:
                            feat = fused_feature[0, :feat_len, :].detach().cpu()
                        
                        feat_norm = torch.norm(feat, dim=1).numpy()
                        feat_img = feat_norm.reshape(H, W)
                        feat_img = cv2.resize(feat_img, (200, 200))
                        feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min() + 1e-9)
                        
                        ax.imshow(feat_img, cmap='viridis', extent=[0, 1, 1, 0])
                        ax.set_title("Fused Feature Spatial Distribution")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha='center')
            else:
                ax.text(0.5, 0.5, "No Fused Feature", ha='center')
            ax.axis('off')
            
            plt.suptitle(f"Feature Comparison - Step {self.debug_step} Layer {layer_idx}")
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_feature_comparison.png")
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Feature comparison vis error: {e}")

    def visualize_adaptive_weights(self, reference_points, layer_idx, adaptive_weights, spatial_shapes):
        """Visualize adaptive weights distribution"""
        if self.debug_step % 1000 != 0:
            return
        
        try:
            # adaptive_weights: (BS, Num_Query, 2) -> [lidar_weight, view_weight]
            if adaptive_weights.dim() == 3:
                weights = adaptive_weights[0].detach().cpu().numpy()  # (Num_Query, 2)
            else:
                weights = adaptive_weights.detach().cpu().numpy()
            
            lidar_weights = weights[:, 0]
            view_weights = weights[:, 1]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot 1: Weight distribution histogram
            ax = axes[0]
            ax.hist(lidar_weights, bins=50, alpha=0.6, label='Lidar Weight', color='green', density=True)
            ax.hist(view_weights, bins=50, alpha=0.6, label='View Weight', color='red', density=True)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.set_title('Adaptive Weights Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Lidar vs View weights scatter
            ax = axes[1]
            ax.scatter(lidar_weights, view_weights, alpha=0.3, s=1, c='blue')
            ax.plot([0, 1], [1, 0], 'r--', linewidth=2, label='lidar + view = 1')
            ax.set_xlabel('Lidar Weight')
            ax.set_ylabel('View Weight')
            ax.set_title('Lidar vs View Weights')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Plot 3: Spatial distribution of weights
            ax = axes[2]
            if spatial_shapes is not None and reference_points is not None:
                pts = reference_points[0].detach().cpu().numpy()  # (Num_Query, 2)
                # Color by lidar weight
                scatter = ax.scatter(pts[:, 0], pts[:, 1], c=lidar_weights, 
                                   cmap='RdYlGn', s=10, alpha=0.6, vmin=0, vmax=1)
                plt.colorbar(scatter, ax=ax, label='Lidar Weight')
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_title('Spatial Distribution of Lidar Weights')
            else:
                ax.text(0.5, 0.5, "No spatial info", ha='center')
            ax.axis('off')
            
            plt.suptitle(f"Adaptive Weights - Step {self.debug_step} Layer {layer_idx}")
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_adaptive_weights.png")
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Adaptive weights vis error: {e}")

    def visualize_cross_attention(self, reference_points, layer_idx, 
                                  query, value_lidar, value_view, spatial_shapes):
        """Visualize cross-attention patterns between query and Lidar/View features"""
        if self.debug_step % 1000 != 0:
            return
        
        try:
            # Compute attention approximation
            def compute_attention_approx(q, v):
                """Compute approximate attention weights"""
                if q.dim() == 3:
                    if q.shape[0] > q.shape[1] * 10 and q.shape[1] < 10:
                        q = q.permute(1, 0, 2)
                if v.dim() == 3:
                    if v.shape[0] > v.shape[1] * 10 and v.shape[1] < 10:
                        v = v.permute(1, 0, 2)
                
                if q.shape[0] != v.shape[0]:
                    min_bs = min(q.shape[0], v.shape[0])
                    q = q[:min_bs]
                    v = v[:min_bs]
                
                q_norm = F.normalize(q, p=2, dim=-1)
                v_norm = F.normalize(v, p=2, dim=-1)
                attn = torch.bmm(q_norm, v_norm.transpose(1, 2))
                attn = F.softmax(attn, dim=-1)
                return attn
            
            if query is not None and value_lidar is not None:
                attn_lidar = compute_attention_approx(query, value_lidar)
                attn_lidar_mean = attn_lidar.mean(dim=0).detach().cpu().numpy()  # (Num_Query, Num_Keys)
            else:
                attn_lidar_mean = None
                
            if query is not None and value_view is not None:
                attn_view = compute_attention_approx(query, value_view)
                attn_view_mean = attn_view.mean(dim=0).detach().cpu().numpy()
            else:
                attn_view_mean = None
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Lidar attention
            ax = axes[0]
            if attn_lidar_mean is not None:
                im = ax.imshow(attn_lidar_mean, cmap='hot', aspect='auto')
                plt.colorbar(im, ax=ax, label='Attention Weight')
                ax.set_xlabel('Lidar Feature Keys')
                ax.set_ylabel('Query')
                ax.set_title('Query-to-Lidar Attention')
            else:
                ax.text(0.5, 0.5, "No Lidar Attention", ha='center')
            
            # Plot 2: View attention
            ax = axes[1]
            if attn_view_mean is not None:
                im = ax.imshow(attn_view_mean, cmap='hot', aspect='auto')
                plt.colorbar(im, ax=ax, label='Attention Weight')
                ax.set_xlabel('View Feature Keys')
                ax.set_ylabel('Query')
                ax.set_title('Query-to-View Attention')
            else:
                ax.text(0.5, 0.5, "No View Attention", ha='center')
            
            plt.suptitle(f"Cross-Attention - Step {self.debug_step} Layer {layer_idx}")
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_{layer_idx}_cross_attention.png")
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Cross-attention vis error: {e}")

    def visualize_layer_progression(self, intermediate_refs, gt_bboxes_3d):
        """Visualize how predictions progress through layers"""
        if self.debug_step % 1000 != 0:
            return
        
        try:
            if intermediate_refs is None or len(intermediate_refs) == 0:
                return
            
            num_layers = len(intermediate_refs)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Compute GT distances for each layer
            gt_distances = []
            for lid in range(num_layers):
                refs = intermediate_refs[lid]  # (BS, Num_Query, 2)
                pts = refs[0].detach().cpu().numpy()
                
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
                                try:
                                    from scipy.spatial.distance import cdist
                                    distances = cdist(pts, gt_pts)
                                    mean_dist = distances.min(axis=1).mean()
                                except ImportError:
                                    # Fallback: simple euclidean distance
                                    distances = np.sqrt(((pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=-1))
                                    mean_dist = distances.min(axis=1).mean()
                                except Exception:
                                    mean_dist = None
                            else:
                                mean_dist = None
                        else:
                            mean_dist = None
                    except:
                        mean_dist = None
                else:
                    mean_dist = None
                
                gt_distances.append(mean_dist)
                
                # Plot predictions for this layer
                if lid < len(axes):
                    ax = axes[lid]
                    ax.scatter(pts[:, 0], pts[:, 1], c='red', s=5, alpha=0.6, label='Pred')
                    
                    if gt_bboxes_3d is not None:
                        try:
                            gt = gt_bboxes_3d[0]
                            if hasattr(gt, 'instance_list'):
                                for line in gt.instance_list:
                                    l_pts = np.array(line.coords)
                                    if hasattr(gt, 'max_x') and hasattr(gt, 'max_y'):
                                        l_pts[:, 0] /= gt.max_x
                                        l_pts[:, 1] /= gt.max_y
                                    ax.plot(l_pts[:, 0], l_pts[:, 1], 'g-', linewidth=1, alpha=0.6)
                        except:
                            pass
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(1, 0)
                    title = f"Layer {lid}"
                    if mean_dist is not None:
                        title += f" (GT Dist: {mean_dist:.4f})"
                    ax.set_title(title)
                    ax.axis('off')
            
            # Summary plot: GT distance vs layer
            if len(axes) > num_layers:
                ax = axes[num_layers]
                if any(d is not None for d in gt_distances):
                    valid_dists = [(i, d) for i, d in enumerate(gt_distances) if d is not None]
                    if len(valid_dists) > 0:
                        layers, dists = zip(*valid_dists)
                        ax.plot(layers, dists, 'b-o', linewidth=2, markersize=8)
                        ax.set_xlabel('Layer')
                        ax.set_ylabel('Mean GT Distance')
                        ax.set_title('Prediction Accuracy vs Layer')
                        ax.grid(True, alpha=0.3)
                        ax.set_xticks(range(num_layers))
            
            plt.suptitle(f"Layer Progression - Step {self.debug_step}")
            plt.tight_layout()
            
            save_path = os.path.join(self.debug_dir, f"step_{self.debug_step}_layer_progression.png")
            plt.savefig(save_path)
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
        
        # [VIS] Save query for visualization
        self._last_query = query
        
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

            refined_view = None # 初始化 refined_view
            gate_values = None
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
                    # [VIS] Save for visualization
                    self._last_adaptive_weights = weights  # (bs, 1, 2)
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
                     fused_feature=current_value if value_view is not None else None,
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
            # Visualize layer progression
            if self.return_intermediate and len(intermediate_reference_points) > 0:
                self.visualize_layer_progression(
                    intermediate_reference_points,
                    kwargs.get('gt_bboxes_3d', None)
                )
            
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
