find_unused_parameters = True
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './new_data_train_result/new_data_categary_split_dataset_pt25'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
bev_res = 0.05
crop_size = (1000, 1000)
input_shape = (1000, 1000)
img_norm_cfg = dict(
    mean_lidar_map=[0.0442, 0.0919, 0.547],
    std_lidar_map=[0.12, 0.1768, 0.1239],
    mean_view_map=[0.2452, 0.2565, 0.2354],
    std_view_map=[0.25739855, 0.26731403, 0.24942544],
    mean_single=[9.1407],
    std_single=[33.3963],
    to_rgb=True)
map_classes = ['lane_line', 'curb', 'stop_line']
attrs_dict = dict(
    laneline_linetype=[
        'single_solid', 'single_dash', 'double_left_solid',
        'double_right_solid', 'double_solid', 'double_dash',
        'colored_three_line', 'other', 'unknown', 'no'
    ],
    laneline_function=[
        'normal', 'fishbone_line', 'diversion_line', 'guide_line', 'other',
        'unknown', 'no'
    ],
    laneline_color=['white', 'yellow', 'other', 'unknown', 'no'],
    curb_linetype=[
        'road_edge', 'plain_edge', 'cone_edge', 'waterhorse_edge',
        'fence_edge', 'park_edge', 'other', 'unknown'
    ],
    stopline_linetype=['normal', 'other', 'unknown'])
attr_classes_num = [10, 7, 5, 8, 3]
fixed_ptsnum_per_gt_line = 25
fixed_ptsnum_per_pred_line = 25
dynamic_sample_config = dict(
    sample_density=2.0,
    min_sample_points=5,
    max_sample_points=25,
    bev_res=0.05)
eval_use_same_gt_sample_num_flag = True
num_map_classes = 3
_dim_ = 256
_ffn_dim_ = 512
instance_num = 50
task_config = dict(with_lidar=True, with_view=True, with_z=False)
input_modality = dict(use_lidar_map=True, use_view_map=True, use_z_map=False)
img_keys = ['lidar_map', 'view_map']
model = dict(
    type='VMA',
    task_config=dict(with_lidar=True, with_view=True, with_z=False),
    modality=dict(use_lidar_map=True, use_view_map=True, use_z_map=False),
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=None,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='/homes/zhangzijian/vma-dev/ckpts/resnet101-5d3b4d8f.pth'),
    img_backbone_view=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='/homes/zhangzijian/vma-dev/ckpts/resnet101-5d3b4d8f.pth'),
    img_neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    img_neck_view=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    pts_bbox_head=dict(
        type='VMAHead',
        num_query=900,
        num_vec=50,
        instance_embed_dim=256,
        num_pts_per_vec=25,
        num_pts_per_gt_vec=25,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        num_classes=3,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='VMADetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention', embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        attr_head=dict(
            type='Attribute_Classifier',
            attr_classes=[10, 7, 5, 8, 3],
            mlp_layers=3,
            in_channels=256,
            feedforward_channels=256),
        bbox_coder=dict(
            type='VMANMSFreeCoder',
            max_num=50,
            score_threshold=0.3,
            num_classes=3),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_attr=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts2lines=dict(type='Pts2LinesLoss', loss_weight=13),
        loss_pts2pts=dict(type='PtsL1Loss', loss_weight=2.5),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.01),
        loss_offset=dict(type='LateralOffsetLoss', loss_weight=3.5)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            out_size_factor=4,
            assigner=dict(
                type='VMAHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=5)))))
dataset_type = 'TrunkLineDataset'
data_root1 = dict(
    train='/homes/zhangzijian/vma-dev/data_1209_merged_split/train/',
    val='/homes/zhangzijian/vma-dev/data_1209_merged_split/val/',
    test='/homes/zhangzijian/vma-dev/data_1209_merged_split/test/')
data_root2 = dict(
    train='/homes/zhangzijian/vma-dev/new_data/',
    val='/homes/zhangzijian/vma-dev/new_data_val/',
    test='/homes/zhangzijian/vma-dev/new_data_val/')
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiImageFromFiles', to_float32=True),
    dict(
        type='CropResizeFlipImage',
        img_keys=['lidar_map', 'view_map'],
        prob=0.2,
        resize_shape=(1000, 1000),
        crop_ratio=(0.8, 1),
        keep_crop_region=False,
        rand_hflip=True),
    dict(
        type='AffineTransform',
        img_keys=['lidar_map', 'view_map'],
        prob=0.2,
        max_rotate_angle=20,
        max_translation_ratio=0.2,
        scale_range=(0.95, 1.25)),
    dict(
        type='NormalizeMultiImage',
        img_keys=['lidar_map', 'view_map'],
        mean_lidar_map=[0.0442, 0.0919, 0.547],
        std_lidar_map=[0.12, 0.1768, 0.1239],
        mean_view_map=[0.2452, 0.2565, 0.2354],
        std_view_map=[0.25739855, 0.26731403, 0.24942544],
        mean_single=[9.1407],
        std_single=[33.3963],
        to_rgb=True),
    dict(
        type='VectorizeMap',
        input_shape=(1000, 1000),
        map_classes=['lane_line', 'curb', 'stop_line'],
        attrs_dict=dict(
            laneline_linetype=[
                'single_solid', 'single_dash', 'double_left_solid',
                'double_right_solid', 'double_solid', 'double_dash',
                'colored_three_line', 'other', 'unknown', 'no'
            ],
            laneline_function=[
                'normal', 'fishbone_line', 'diversion_line', 'guide_line',
                'other', 'unknown', 'no'
            ],
            laneline_color=['white', 'yellow', 'other', 'unknown', 'no'],
            curb_linetype=[
                'road_edge', 'plain_edge', 'cone_edge', 'waterhorse_edge',
                'fence_edge', 'park_edge', 'other', 'unknown'
            ],
            stopline_linetype=['normal', 'other', 'unknown']),
        fixed_ptsnum_per_line=25),
    dict(type='TrunkVMAAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_attrs', 'img_keys',
            'img_metas'
        ],
        meta_keys=[])
]
test_pipeline = [
    dict(type='LoadMultiImageFromFiles', to_float32=True),
    dict(
        type='NormalizeMultiImage',
        img_keys=['lidar_map', 'view_map'],
        mean_lidar_map=[0.0442, 0.0919, 0.547],
        std_lidar_map=[0.12, 0.1768, 0.1239],
        mean_view_map=[0.2452, 0.2565, 0.2354],
        std_view_map=[0.25739855, 0.26731403, 0.24942544],
        mean_single=[9.1407],
        std_single=[33.3963],
        to_rgb=True),
    dict(
        type='VectorizeMap',
        input_shape=(1000, 1000),
        map_classes=['lane_line', 'curb', 'stop_line'],
        attrs_dict=dict(
            laneline_linetype=[
                'single_solid', 'single_dash', 'double_left_solid',
                'double_right_solid', 'double_solid', 'double_dash',
                'colored_three_line', 'other', 'unknown', 'no'
            ],
            laneline_function=[
                'normal', 'fishbone_line', 'diversion_line', 'guide_line',
                'other', 'unknown', 'no'
            ],
            laneline_color=['white', 'yellow', 'other', 'unknown', 'no'],
            curb_linetype=[
                'road_edge', 'plain_edge', 'cone_edge', 'waterhorse_edge',
                'fence_edge', 'park_edge', 'other', 'unknown'
            ],
            stopline_linetype=['normal', 'other', 'unknown']),
        fixed_ptsnum_per_line=25),
    dict(type='TrunkVMAAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_attrs', 'img_keys',
            'img_metas'
        ],
        meta_keys=[])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='TrunkLineDataset',
                data_root=
                '/homes/zhangzijian/vma-dev/data_1209_merged_split/train/',
                imgs_dir='cropped_data/images',
                annots_dir='cropped_data/annots',
                mask_dir=None,
                points_nums=25,
                modality=dict(
                    use_lidar_map=True, use_view_map=True, use_z_map=False),
                view_img_dir='cropped_data/view_maps',
                z_map_dir='cropped_data/z_maps_vis',
                map_classes=['lane_line', 'curb', 'stop_line'],
                attrs_dict=dict(
                    laneline_linetype=[
                        'single_solid', 'single_dash', 'double_left_solid',
                        'double_right_solid', 'double_solid', 'double_dash',
                        'colored_three_line', 'other', 'unknown', 'no'
                    ],
                    laneline_function=[
                        'normal', 'fishbone_line', 'diversion_line',
                        'guide_line', 'other', 'unknown', 'no'
                    ],
                    laneline_color=[
                        'white', 'yellow', 'other', 'unknown', 'no'
                    ],
                    curb_linetype=[
                        'road_edge', 'plain_edge', 'cone_edge',
                        'waterhorse_edge', 'fence_edge', 'park_edge', 'other',
                        'unknown'
                    ],
                    stopline_linetype=['normal', 'other', 'unknown']),
                eval_use_same_gt_sample_num_flag=True,
                dynamic_sample_config=dict(
                    sample_density=2.0,
                    min_sample_points=5,
                    max_sample_points=25,
                    bev_res=0.05),
                pipeline=[
                    dict(type='LoadMultiImageFromFiles', to_float32=True),
                    dict(
                        type='CropResizeFlipImage',
                        img_keys=['lidar_map', 'view_map'],
                        prob=0.2,
                        resize_shape=(1000, 1000),
                        crop_ratio=(0.8, 1),
                        keep_crop_region=False,
                        rand_hflip=True),
                    dict(
                        type='AffineTransform',
                        img_keys=['lidar_map', 'view_map'],
                        prob=0.2,
                        max_rotate_angle=20,
                        max_translation_ratio=0.2,
                        scale_range=(0.95, 1.25)),
                    dict(
                        type='NormalizeMultiImage',
                        img_keys=['lidar_map', 'view_map'],
                        mean_lidar_map=[0.0442, 0.0919, 0.547],
                        std_lidar_map=[0.12, 0.1768, 0.1239],
                        mean_view_map=[0.2452, 0.2565, 0.2354],
                        std_view_map=[0.25739855, 0.26731403, 0.24942544],
                        mean_single=[9.1407],
                        std_single=[33.3963],
                        to_rgb=True),
                    dict(
                        type='VectorizeMap',
                        input_shape=(1000, 1000),
                        map_classes=['lane_line', 'curb', 'stop_line'],
                        attrs_dict=dict(
                            laneline_linetype=[
                                'single_solid', 'single_dash',
                                'double_left_solid', 'double_right_solid',
                                'double_solid', 'double_dash',
                                'colored_three_line', 'other', 'unknown', 'no'
                            ],
                            laneline_function=[
                                'normal', 'fishbone_line', 'diversion_line',
                                'guide_line', 'other', 'unknown', 'no'
                            ],
                            laneline_color=[
                                'white', 'yellow', 'other', 'unknown', 'no'
                            ],
                            curb_linetype=[
                                'road_edge', 'plain_edge', 'cone_edge',
                                'waterhorse_edge', 'fence_edge', 'park_edge',
                                'other', 'unknown'
                            ],
                            stopline_linetype=['normal', 'other', 'unknown']),
                        fixed_ptsnum_per_line=25),
                    dict(type='TrunkVMAAdaptor'),
                    dict(
                        type='Collect',
                        keys=[
                            'img', 'gt_bboxes', 'gt_labels', 'gt_attrs',
                            'img_keys', 'img_metas'
                        ],
                        meta_keys=[])
                ],
                mode='train',
                test_mode=False),
            dict(
                type='TrunkLineDataset',
                data_root='/homes/zhangzijian/vma-dev/new_data/',
                imgs_dir='cropped_data/images',
                annots_dir='cropped_data/annots',
                mask_dir=None,
                points_nums=25,
                modality=dict(
                    use_lidar_map=True, use_view_map=True, use_z_map=False),
                view_img_dir='cropped_data/view_maps',
                z_map_dir='cropped_data/z_maps_vis',
                map_classes=['lane_line', 'curb', 'stop_line'],
                attrs_dict=dict(
                    laneline_linetype=[
                        'single_solid', 'single_dash', 'double_left_solid',
                        'double_right_solid', 'double_solid', 'double_dash',
                        'colored_three_line', 'other', 'unknown', 'no'
                    ],
                    laneline_function=[
                        'normal', 'fishbone_line', 'diversion_line',
                        'guide_line', 'other', 'unknown', 'no'
                    ],
                    laneline_color=[
                        'white', 'yellow', 'other', 'unknown', 'no'
                    ],
                    curb_linetype=[
                        'road_edge', 'plain_edge', 'cone_edge',
                        'waterhorse_edge', 'fence_edge', 'park_edge', 'other',
                        'unknown'
                    ],
                    stopline_linetype=['normal', 'other', 'unknown']),
                eval_use_same_gt_sample_num_flag=True,
                dynamic_sample_config=dict(
                    sample_density=2.0,
                    min_sample_points=5,
                    max_sample_points=25,
                    bev_res=0.05),
                pipeline=[
                    dict(type='LoadMultiImageFromFiles', to_float32=True),
                    dict(
                        type='CropResizeFlipImage',
                        img_keys=['lidar_map', 'view_map'],
                        prob=0.2,
                        resize_shape=(1000, 1000),
                        crop_ratio=(0.8, 1),
                        keep_crop_region=False,
                        rand_hflip=True),
                    dict(
                        type='AffineTransform',
                        img_keys=['lidar_map', 'view_map'],
                        prob=0.2,
                        max_rotate_angle=20,
                        max_translation_ratio=0.2,
                        scale_range=(0.95, 1.25)),
                    dict(
                        type='NormalizeMultiImage',
                        img_keys=['lidar_map', 'view_map'],
                        mean_lidar_map=[0.0442, 0.0919, 0.547],
                        std_lidar_map=[0.12, 0.1768, 0.1239],
                        mean_view_map=[0.2452, 0.2565, 0.2354],
                        std_view_map=[0.25739855, 0.26731403, 0.24942544],
                        mean_single=[9.1407],
                        std_single=[33.3963],
                        to_rgb=True),
                    dict(
                        type='VectorizeMap',
                        input_shape=(1000, 1000),
                        map_classes=['lane_line', 'curb', 'stop_line'],
                        attrs_dict=dict(
                            laneline_linetype=[
                                'single_solid', 'single_dash',
                                'double_left_solid', 'double_right_solid',
                                'double_solid', 'double_dash',
                                'colored_three_line', 'other', 'unknown', 'no'
                            ],
                            laneline_function=[
                                'normal', 'fishbone_line', 'diversion_line',
                                'guide_line', 'other', 'unknown', 'no'
                            ],
                            laneline_color=[
                                'white', 'yellow', 'other', 'unknown', 'no'
                            ],
                            curb_linetype=[
                                'road_edge', 'plain_edge', 'cone_edge',
                                'waterhorse_edge', 'fence_edge', 'park_edge',
                                'other', 'unknown'
                            ],
                            stopline_linetype=['normal', 'other', 'unknown']),
                        fixed_ptsnum_per_line=25),
                    dict(type='TrunkVMAAdaptor'),
                    dict(
                        type='Collect',
                        keys=[
                            'img', 'gt_bboxes', 'gt_labels', 'gt_attrs',
                            'img_keys', 'img_metas'
                        ],
                        meta_keys=[])
                ],
                mode='train',
                test_mode=False)
        ],
        separate_eval=False),
    val=dict(
        type='TrunkLineDataset',
        data_root='/homes/zhangzijian/vma-dev/data_1209_merged_split/val/',
        imgs_dir='cropped_data/images',
        annots_dir='cropped_data/annots',
        mask_dir=None,
        points_nums=25,
        modality=dict(use_lidar_map=True, use_view_map=True, use_z_map=False),
        view_img_dir='cropped_data/view_maps',
        z_map_dir='cropped_data/z_maps_vis',
        map_classes=['lane_line', 'curb', 'stop_line'],
        attrs_dict=dict(
            laneline_linetype=[
                'single_solid', 'single_dash', 'double_left_solid',
                'double_right_solid', 'double_solid', 'double_dash',
                'colored_three_line', 'other', 'unknown', 'no'
            ],
            laneline_function=[
                'normal', 'fishbone_line', 'diversion_line', 'guide_line',
                'other', 'unknown', 'no'
            ],
            laneline_color=['white', 'yellow', 'other', 'unknown', 'no'],
            curb_linetype=[
                'road_edge', 'plain_edge', 'cone_edge', 'waterhorse_edge',
                'fence_edge', 'park_edge', 'other', 'unknown'
            ],
            stopline_linetype=['normal', 'other', 'unknown']),
        eval_use_same_gt_sample_num_flag=True,
        dynamic_sample_config=dict(
            sample_density=2.0,
            min_sample_points=5,
            max_sample_points=25,
            bev_res=0.05),
        pipeline=[
            dict(type='LoadMultiImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiImage',
                img_keys=['lidar_map', 'view_map'],
                mean_lidar_map=[0.0442, 0.0919, 0.547],
                std_lidar_map=[0.12, 0.1768, 0.1239],
                mean_view_map=[0.2452, 0.2565, 0.2354],
                std_view_map=[0.25739855, 0.26731403, 0.24942544],
                mean_single=[9.1407],
                std_single=[33.3963],
                to_rgb=True),
            dict(
                type='VectorizeMap',
                input_shape=(1000, 1000),
                map_classes=['lane_line', 'curb', 'stop_line'],
                attrs_dict=dict(
                    laneline_linetype=[
                        'single_solid', 'single_dash', 'double_left_solid',
                        'double_right_solid', 'double_solid', 'double_dash',
                        'colored_three_line', 'other', 'unknown', 'no'
                    ],
                    laneline_function=[
                        'normal', 'fishbone_line', 'diversion_line',
                        'guide_line', 'other', 'unknown', 'no'
                    ],
                    laneline_color=[
                        'white', 'yellow', 'other', 'unknown', 'no'
                    ],
                    curb_linetype=[
                        'road_edge', 'plain_edge', 'cone_edge',
                        'waterhorse_edge', 'fence_edge', 'park_edge', 'other',
                        'unknown'
                    ],
                    stopline_linetype=['normal', 'other', 'unknown']),
                fixed_ptsnum_per_line=25),
            dict(type='TrunkVMAAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_attrs', 'img_keys',
                    'img_metas'
                ],
                meta_keys=[])
        ],
        mode='valid',
        test_mode=False),
    test=dict(
        type='TrunkLineDataset',
        data_root='/homes/zhangzijian/vma-dev/data_1209_merged_split/test/',
        imgs_dir='cropped_data/images',
        annots_dir='cropped_data/annots',
        mask_dir=None,
        points_nums=25,
        modality=dict(use_lidar_map=True, use_view_map=True, use_z_map=False),
        view_img_dir='cropped_data/view_maps',
        z_map_dir='cropped_data/z_maps_vis',
        map_classes=['lane_line', 'curb', 'stop_line'],
        attrs_dict=dict(
            laneline_linetype=[
                'single_solid', 'single_dash', 'double_left_solid',
                'double_right_solid', 'double_solid', 'double_dash',
                'colored_three_line', 'other', 'unknown', 'no'
            ],
            laneline_function=[
                'normal', 'fishbone_line', 'diversion_line', 'guide_line',
                'other', 'unknown', 'no'
            ],
            laneline_color=['white', 'yellow', 'other', 'unknown', 'no'],
            curb_linetype=[
                'road_edge', 'plain_edge', 'cone_edge', 'waterhorse_edge',
                'fence_edge', 'park_edge', 'other', 'unknown'
            ],
            stopline_linetype=['normal', 'other', 'unknown']),
        eval_use_same_gt_sample_num_flag=True,
        dynamic_sample_config=dict(
            sample_density=2.0,
            min_sample_points=5,
            max_sample_points=25,
            bev_res=0.05),
        pipeline=[
            dict(type='LoadMultiImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiImage',
                img_keys=['lidar_map', 'view_map'],
                mean_lidar_map=[0.0442, 0.0919, 0.547],
                std_lidar_map=[0.12, 0.1768, 0.1239],
                mean_view_map=[0.2452, 0.2565, 0.2354],
                std_view_map=[0.25739855, 0.26731403, 0.24942544],
                mean_single=[9.1407],
                std_single=[33.3963],
                to_rgb=True),
            dict(
                type='VectorizeMap',
                input_shape=(1000, 1000),
                map_classes=['lane_line', 'curb', 'stop_line'],
                attrs_dict=dict(
                    laneline_linetype=[
                        'single_solid', 'single_dash', 'double_left_solid',
                        'double_right_solid', 'double_solid', 'double_dash',
                        'colored_three_line', 'other', 'unknown', 'no'
                    ],
                    laneline_function=[
                        'normal', 'fishbone_line', 'diversion_line',
                        'guide_line', 'other', 'unknown', 'no'
                    ],
                    laneline_color=[
                        'white', 'yellow', 'other', 'unknown', 'no'
                    ],
                    curb_linetype=[
                        'road_edge', 'plain_edge', 'cone_edge',
                        'waterhorse_edge', 'fence_edge', 'park_edge', 'other',
                        'unknown'
                    ],
                    stopline_linetype=['normal', 'other', 'unknown']),
                fixed_ptsnum_per_line=25),
            dict(type='TrunkVMAAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_attrs', 'img_keys',
                    'img_metas'
                ],
                meta_keys=[])
        ],
        mode='test',
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
optimizer = dict(
    type='AdamW',
    lr=5e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.1),
            img_backbone_view=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 200
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=8)
evaluation = dict(interval=4, metric='chamfer')
gpu_ids = range(0, 1)
