from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.vma.modules import *


@DETECTORS.register_module()
class VMA(MVXTwoStageDetector):  #添加FUSE_ENCODER

    def __init__(self,
                 task_config=None,
                 modality=None,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fuse_encoder=None,
                 fuse_neck=None,
                 ):

        super(VMA,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        if task_config is None:
            self.task_config = dict(
                with_lidar=True,
                with_view=False,
                with_z=False,
            )
        if modality is None:
            self.moality = dict(
                use_lidar_map=True,
                use_view_map=False,
                use_z_map=False,
            )
        else:
            self.moality = modality  
        
        # 添加DDP兼容标志
        self._ddp_params_and_buffers_to_ignore = []
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        if fuse_neck is not None:
            self.fuse_neck = build_fuse_neck(fuse_neck)
            self.fuse_encoder = None
        else:
            self.fuse_neck = None
            if fuse_encoder is not None:
                self.fuse_encoder = build_fuse_encoder(fuse_encoder)
            else:
                self.fuse_encoder = None

    def extract_img_feat(self, 
                         imgs, 
                        #  view_img=None  #添加VIEW
                         img_keys,
                         img_metas
                         ):
        """Extract features of images."""
        B = imgs.size(0)
        if imgs is not None:
            
            if self.moality["use_lidar_map"]:
                img_idx = img_keys[0].index("lidar_map")
                img = imgs[:, img_idx]

                if img.dim() == 5 and img.size(0) == 1:
                    img.squeeze_()
                elif img.dim() == 5 and img.size(0) > 1:
                    B, N, C, H, W = img.size()
                    img = img.reshape(B * N, C, H, W)
            
            if self.moality["use_view_map"]:
                img_idx = img_keys[0]["img_keys"].index("lidar_map")
                view_img = imgs[:, img_idx]
                
                if view_img.dim() == 5 and view_img.size(0) == 1:
                    view_img.squeeze_()
                elif view_img.dim() == 5 and view_img.size(0) > 1:
                    B, N, C, H, W = view_img.size()
                    view_img = view_img.reshape(B * N, C, H, W)
                
                if self.fuse_neck is None:
                    # 融合两种图像
                    if self.fuse_encoder is not None:
                        img = self.fuse_encoder(img, view_img)

            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            
            if self.fuse_neck is not None:
                if self.use_grid_mask:
                    view_img = self.grid_mask(view_img)

                view_img_feats = self.img_backbone(view_img)
                if isinstance(view_img_feats, dict):
                    view_img_feats = list(view_img_feats.values())
                
                img_feats = self.fuse_neck(img_feats,view_img_feats)

        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        return img_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, 
                     img,
                     img_keys,
                     img_metas=None, 
                     ):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_keys, img_metas)
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_labels, 
                          gt_bboxes,
                          gt_attrs,
                          img_metas,
                          ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_labels (list[torch.Tensor]): Ground truth labels for
                boxes of each sample
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes (list[torch.Tensor], optional): Ground truth
                boxes.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_labels, gt_bboxes, gt_attrs, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas)
        return losses

    def forward_dummy(self, img, view_img=None):
        dummy_metas = None
        return self.forward_test(img=img, view_img=view_img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      img=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_attrs=None,
                      img_keys=None,
                      img_metas=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats = self.extract_feat(img=img, img_keys=img_keys)
        if torch.isnan(img).any() or (not torch.isfinite(img).all()):
            print("Tensor contains NaN values")
            import pdb; pdb.set_trace()
        for img_feat in img_feats:
            if torch.isnan(img_feat).any() or (not torch.isfinite(img_feat).all()):
                print("Tensor contains NaN values")
                import pdb; pdb.set_trace()
        
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_labels, gt_bboxes, gt_attrs, img_metas)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                    img_metas=None,
                    img=None,
                    img_keys=None,
                    rescale=None,
                    **kwargs,
                    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        bbox_results = self.simple_test(
            img_metas, img, img_keys, **kwargs)
        for idx in range(len(bbox_results)):
            bbox_results[idx]['pts_bbox']['img_metas'] = img_metas[idx]
            for key, value in kwargs.items():
                bbox_results[idx]['pts_bbox'][key] = value[idx]
        return bbox_results

    def pred2result(self, 
                    scores, 
                    labels, 
                    pts, 
                    attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            pts (torch.Tensor): Points with shape of (n, 2).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs

        return result_dict
    
    def simple_test_pts(self, 
                        x, 
                        img_metas, 
                        ):
        """Test function"""
        
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, 
                                                  img_metas, 
                                                  )
        bbox_results = [
            self.pred2result(scores, labels, pts, attrs)
            for scores, labels, pts, attrs in bbox_list
        ]
        return bbox_results
    def simple_test(self, 
                    img_metas, 
                    img=None, 
                    img_keys=None,
                    **kwargs
                    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_keys=img_keys, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list