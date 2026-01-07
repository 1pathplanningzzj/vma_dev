from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.vma.modules import *  # 确保包含view交叉注意力相关模块
from mmdet3d.models.builder import build_backbone, build_neck, build_head
from projects.mmdet3d_plugin.vma.modules.quality_estimator import ViewQualityEstimator

@DETECTORS.register_module()
class VMA(MVXTwoStageDetector):  

    def __init__(self,
                 task_config=None,
                 modality=None,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 img_backbone_view=None,
                 pts_backbone=None,
                 img_neck=None,
                 img_neck_view=None,
                 img_neck_lidar=None,
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

        super(VMA, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)
        
        # 任务配置与模态设置
        self.task_config = task_config if task_config is not None else dict(
            with_lidar=True, with_view=False, with_z=False)
        self.modality = modality if modality is not None else dict(
            use_lidar_map=True, use_view_map=False, use_z_map=False)
        
        # 网格掩码（数据增强）
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode

        # 构建view分支的backbone和neck（双分支并行）
        if img_backbone_view is not None:
            self.img_backbone_view = build_backbone(img_backbone_view)
        else:
            self.img_backbone_view = None
        if img_neck_view is not None:
            self.img_neck_view = build_neck(img_neck_view)
        else:
            self.img_neck_view = None

        # self.fuse_neck = build_neck(fuse_neck) if fuse_neck is not None else None
        # self.fuse_encoder = build_backbone(fuse_encoder) if fuse_encoder is not None else None

    def extract_img_feat(self, 
                        imgs, 
                        img_keys,
                        img_metas,
                        view_img=None,
                        z_map_img=None,
                        ):
        """提取双分支图像特征（lidar_map和view_map）"""
        B = imgs.size(0) if imgs is not None else 0
        branch_feats = dict(lidar=None, view=None)  # 存储双分支特征

        if imgs is not None:
            # 处理图像维度（单帧图像去冗余维度）
            if imgs.dim() == 5 and imgs.size(1) == 1:
                imgs = imgs.squeeze(1)  # [B, 1, C, H, W] → [B, C, H, W]

            # 拆分lidar_map（前3通道）和view_map（后3通道）
            # 注意：需与数据加载时的通道拼接逻辑一致
            if self.modality["use_lidar_map"]:
                lidar_img = imgs[:, 0:3, :, :].contiguous()  # [B, 3, H, W]
            else:
                lidar_img = None
            if self.modality["use_view_map"]:
                view_img = imgs[:, 3:6, :, :].contiguous()  # [B, 3, H, W]
            else:
                view_img = None

            # 1. 提取lidar分支特征（主分支）
            if self.modality["use_lidar_map"] and lidar_img is not None:
                if self.use_grid_mask:
                    lidar_img = self.grid_mask(lidar_img)  # 网格掩码增强
                #  backbone特征提取（如ResNet/Swin）
                lidar_backbone_feats = self.img_backbone(lidar_img)
                # 转换为列表格式（适配neck输入）
                if isinstance(lidar_backbone_feats, dict):
                    lidar_backbone_feats = list(lidar_backbone_feats.values())
                # neck处理（如ChannelMapper，输出多尺度特征）
                if self.img_neck is not None:
                    branch_feats['lidar'] = self.img_neck(lidar_backbone_feats)

            # 2. 提取view分支特征（辅助分支）
            if self.modality["use_view_map"] and view_img is not None and self.img_backbone_view is not None:
                if self.use_grid_mask:
                    view_img = self.grid_mask(view_img)  # 网格掩码增强
                # backbone特征提取（如ResNet）
                view_backbone_feats = self.img_backbone_view(view_img)
                # 转换为列表格式（适配neck输入）
                if isinstance(view_backbone_feats, dict):
                    view_backbone_feats = list(view_backbone_feats.values())
                # neck处理（如ChannelMapper，输出多尺度特征）
                if self.img_neck_view is not None:
                    branch_feats['view'] = self.img_neck_view(view_backbone_feats)
                # view_high_level_feat = [view_backbone_feats[-1]]  # 用列表包裹，保持与多尺度格式一致（长度为1）
                # branch_feats['view'] = view_high_level_feat  # 此时view_feats是[Tensor(bs, 2048, H, W)]
            return branch_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, 
                     img,
                     img_keys,
                     img_metas=None, 
                     ):
        """提取图像特征入口（返回双分支特征）"""
        return self.extract_img_feat(img, img_keys, img_metas)

    def forward_pts_train(self,
                        lidar_feats,  # lidar分支多尺度特征（列表）
                        view_feats,   # view分支多尺度特征（列表）
                        gt_labels, 
                        gt_bboxes,
                        gt_attrs,
                        img_metas,
                        ):
        """点云分支训练前向（传递双分支特征到Head）"""
        # 调用VMAHead.forward，传入双分支特征
        outs = self.pts_bbox_head(
            mlvl_feats=lidar_feats,       # lidar分支多尺度特征
            mlvl_feats_view=view_feats,   # view分支多尺度特征
            img_metas=img_metas
        )
        # 计算损失
        loss_inputs = [gt_labels, gt_bboxes, gt_attrs, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas)
        return losses

    def forward_dummy(self, img, view_img=None, z_map_img=None):
        """虚拟前向（用于测试网络结构）"""
        # 修复原代码中z_map_img未定义的问题
        dummy_metas = [dict(img_shape=img.shape[2:])]  # 构造虚拟meta信息
        return self.forward_test(
            img=img, 
            img_metas=[dummy_metas], 
            img_keys=["lidar_map", "view_map"]
        )

    def forward(self, return_loss=True, **kwargs):
        """统一前向入口（训练/测试分支）"""
        if return_loss:
            return self.forward_train(** kwargs)
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
        branch_feats = self.extract_feat(img=img, img_keys=img_keys)
        lidar_feats = branch_feats['lidar']
        view_feats = branch_feats['view']

        # img_feats = self.extract_feat(img=img, img_keys=img_keys)
        
        losses = dict()
        losses_pts = self.forward_pts_train(
            lidar_feats=lidar_feats,
            view_feats=view_feats,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            gt_attrs=gt_attrs,
            img_metas=img_metas
        )
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
                        lidar_feats,  # lidar分支特征
                        view_feats,   # view分支特征
                        img_metas, 
                        ):
        """Test function（适配双分支特征）"""
        # 传入双分支特征到Head
        outs = self.pts_bbox_head(
            mlvl_feats=lidar_feats,
            mlvl_feats_view=view_feats,
            img_metas=img_metas
        )
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
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
        """Test function without augmentation（修复特征传递）"""
        # 1. 处理img形态：若img是列表，转为张量（避免列表无size()的问题）
        if img is not None and isinstance(img, list):
            img = torch.stack(img) if len(img) > 1 else img[0].unsqueeze(0)  # 补batch维度

        # 2. 提取双分支特征（字典）
        img_feats = self.extract_feat(img=img, img_keys=img_keys, img_metas=img_metas)
        lidar_feats = img_feats['lidar']
        view_feats = img_feats['view']

        # 3. 检查特征非空
        assert lidar_feats is not None, ""
        if self.modality["use_view_map"]:
            assert view_feats is not None, ""

        # 4. 传入双分支特征到 simple_test_pts
        bbox_pts = self.simple_test_pts(
            lidar_feats=lidar_feats,
            view_feats=view_feats,
            img_metas=img_metas
        )

        # 5. 组装结果
        bbox_list = [dict() for _ in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list