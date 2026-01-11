import torch
import torch.nn as nn
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.transformer import DeformableDetrTransformer, inverse_sigmoid
from .decoder import VMADetectionTransformerDecoder

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class SplitModalityDecoder(VMADetectionTransformerDecoder):
    def __init__(self, *args, split_layer_index=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_layer_index = split_layer_index

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

            # Logic to switch value based on layer index
            current_value = value_lidar
            if lid >= self.split_layer_index and value_view is not None:
                current_value = value_view
            
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

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class SplitModalityTransformer(DeformableDetrTransformer):
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
                if self.num_feature_levels > 1:
                    feat = feat + self.level_embeds[lvl].view(1, 1, -1).to(feat.dtype)
                feat_flatten_view.append(feat)
            memory_view = torch.cat(feat_flatten_view, 1).permute(1, 0, 2)

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
