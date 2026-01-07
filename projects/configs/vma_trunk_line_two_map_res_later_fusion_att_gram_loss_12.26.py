find_unused_parameters = True

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
#  python3 tools/test.py ./projects/configs/vma_trunk_line_two_map_res_later_fusion_att.py  /homes/zhangzijian/vma-dev/new_data_train_result/res_swin_later_fusion_spatial_gram_loss_debug/epoch_48.pth --eval chamfer --eval-option show=True show_dir=three_map_visualize_result_fuse

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = "./new_data_train_result/res101_res34_later_fusion_spatial_gram_loss"
load_from = None
resume_from = None
workflow = [('train', 1)]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
bev_res = 0.05
crop_size = (1000, 1000)
input_shape = (1000, 1000) # (w, h), equal to cropped patch image
# img_norm_cfg = dict(mean_lidar_map =[0.5470, 0.0919, 0.0442], 
#                     std_lidar_map =[0.1239,  0.1768, 0.1200],  
#                     mean_view_map =[0.2354,  0.2565, 0.2452],  
#                     std_view_map = [0.24942544, 0.26731403, 0.25739855],
img_norm_cfg = dict(mean_lidar_map =[0.0442, 0.0919, 0.5470], 
                    std_lidar_map =[0.1200,  0.1768, 0.1239],  
                    mean_view_map =[0.2452,  0.2565, 0.2354],  
                    std_view_map = [0.25739855, 0.26731403, 0.24942544],
                    mean_single=[9.1407],  
                    std_single=[33.3963],  
                    to_rgb=True)

map_classes = ['lane_line', 'curb', 'stop_line']

## trunk label
#laneline
attrs_dict = dict(
    #     laneline_linetype=[ "single_solid", "single_dash", "double_solid", "double_dash", "thick_dash",                                    # "left_wait_line",  "double_soild", "double_right_soild", "double_left_soild", 
    laneline_linetype=[ "single_solid", "single_dash", "double_solid", "double_dash", "thick_dash",
                        "other", "unknown", "no"],
    laneline_shape=["normal", "fishbone", "unknown"],
    # laneline_direction=["same", "opposite_with_curb", "opposite_without_curb", "unknown"],
    curb_linetype=["road_boundary", "cone_boundary", "other", "unknown"],
    stopline_linetype=["normal", "other", "unknown"],
)
attr_classes_num = [len(x) for x in attrs_dict.values()]

fixed_ptsnum_per_gt_line = 50 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 50
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)
_dim_ = 256
_ffn_dim_ = _dim_*2
instance_num=50
'''fuse_encoder=dict(
        type='SimpleFusionEncoder',
        in_channels=6,  # 假设img和view_img都是3通道
        out_channels=3,
        conv_cfg=None,
        norm_cfg=None),'''
'''fuse_neck=dict(
        type='ViewFeatureFusion',
        in_channels=[512, 1024, 2048],
        offset_channels=64),'''

# =========== modality ===============
task_config = dict(
    with_lidar=True,
    with_view=True,
    with_z=False,
)
input_modality = dict(
    use_lidar_map=True,
    use_view_map=True,
    use_z_map=False, # npy file should be customed specially
)
# img_keys=["lidar_map"]

img_keys =["lidar_map", "view_map"]
# img_keys =["lidar_map", "view_map", "z_map"]

model = dict(
    type='VMA',
    task_config=task_config,
    modality=input_modality,
    use_grid_mask=True,
    video_test_mode=False,
    pretrained = None,

    img_backbone=dict(
        type='ResNet',
        depth=101, 
        num_stages=4,
        out_indices=(1,2,3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='ckpts/resnet101-5d3b4d8f.pth'
    ),

    img_backbone_view=dict(
        type='ResNet',
        depth=34, 
        num_stages=4,
        out_indices=(1,2,3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='ckpts/resnet34-333f7ec4.pth'
        ),

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
        in_channels=[128, 256, 512],          
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
        
    pts_bbox_head=dict(
        type='VMAHead',
        num_query=900,
        num_vec=instance_num,
        instance_embed_dim=256,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        num_classes=num_map_classes,
        in_channels=_dim_,
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
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        attr_head=dict(
            type='Attribute_Classifier', 
            attr_classes=attr_classes_num,
            mlp_layers=3,
            in_channels=256, 
            feedforward_channels=256   
        ),
        bbox_coder=dict(
            type='VMANMSFreeCoder',
            max_num=instance_num,
            score_threshold=0.3,
            num_classes=num_map_classes),
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
        loss_pts2lines=dict(type='Pts2LinesLoss', 
                      loss_weight=7.5),
        loss_pts2pts=dict(type='PtsL1Loss', 
                      loss_weight=1.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.01), # 方向损失加大一点
        loss_offset=dict(type='LateralOffsetLoss',loss_weight=2.0),
        # loss_slope=dict(type='MSELoss', loss_weight=0.5), # 增加 斜率损失进行测试 
),
     
   
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        out_size_factor=4,
        assigner=dict(
            type='VMAHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            )
        )
    )
)

# ================ data ========================

dataset_type = 'TrunkLineDataset'
data_root = dict(
    train="new_data/",
    val="new_data_val/",
    test="new_data_val/",
)
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiImageFromFiles', to_float32=True),
    dict(
        type='CropResizeFlipImage',
        img_keys=img_keys,
        prob=0.2,
        resize_shape=input_shape,
        crop_ratio=(0.8, 1),
        keep_crop_region=False,
        rand_hflip=True,
    ),
    dict(
        type='AffineTransform',
        img_keys=img_keys,
        prob=0.2,
        max_rotate_angle=20,
        max_translation_ratio=0.2,
        scale_range=(0.95, 1.25)
    ),

    dict(type='NormalizeMultiImage', img_keys=img_keys, **img_norm_cfg),
    dict(
        type="VectorizeMap",
        input_shape=input_shape,
        map_classes=map_classes,
        attrs_dict=attrs_dict,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    ),
    dict(type="TrunkVMAAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img", 
            "gt_bboxes", 
            "gt_labels", 
            "gt_attrs", 
            "img_keys",
            "img_metas",
            ],
        meta_keys=[],
    )
]

test_pipeline = [
    dict(type='LoadMultiImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiImage', img_keys=img_keys, **img_norm_cfg),
    dict(
        type="VectorizeMap",
        input_shape=input_shape,
        map_classes=map_classes,
        attrs_dict=attrs_dict,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    ),
    dict(type="TrunkVMAAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img", 
            "gt_bboxes", 
            "gt_labels", 
            "gt_attrs", 
            "img_keys",
            "img_metas",
            ],
        meta_keys=[],
    )
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root["train"],
        imgs_dir="cropped_data/images",
        annots_dir="cropped_data/annots",
        mask_dir=None,
        points_nums=fixed_ptsnum_per_gt_line,
        modality=input_modality,
        view_img_dir="cropped_data/view_maps",  #添加VIEW
        z_map_dir = "cropped_data/z_maps_vis",
        map_classes=map_classes,
        attrs_dict=attrs_dict, 
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        pipeline=train_pipeline,
        mode='train',
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root["val"],
        imgs_dir="cropped_data/images",
        annots_dir="cropped_data/annots",
        mask_dir=None,
        points_nums=fixed_ptsnum_per_gt_line,
        modality=input_modality,
        view_img_dir="cropped_data/view_maps",  #添加VIEW
        z_map_dir = "cropped_data/z_maps_vis",

        map_classes=map_classes,
        attrs_dict = attrs_dict, 
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        pipeline=test_pipeline,
        mode='valid',
        test_mode=False),
    test=dict(
        type=dataset_type,
        data_root=data_root["test"],
        imgs_dir="cropped_data/images",
        annots_dir="cropped_data/annots",
        mask_dir=None,
        points_nums=fixed_ptsnum_per_gt_line,
        modality=input_modality,
        view_img_dir="cropped_data/view_maps",  #添加VIEW
        z_map_dir = "cropped_data/z_maps_vis",

        map_classes=map_classes,
        attrs_dict = attrs_dict, 
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        pipeline=test_pipeline,
        mode='test',
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=0.5e-4,

    # lr=0.3e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'img_backbone_view': dict(lr_mult=0.1),  # 新增：view分支骨干降权
        }),
    weight_decay=0.01)

# optimizer_config = dict(grad_clip=dict(max_norm=12, norm_type=2))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
 
total_epochs = 100
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=8)

evaluation = dict(interval=2, metric='chamfer')
# fp16 = dict(loss_scale=512.)
