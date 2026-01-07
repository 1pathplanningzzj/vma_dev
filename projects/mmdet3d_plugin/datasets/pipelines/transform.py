import numpy as np
from numpy import random
import cv2
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
import mmcv
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class NormalizeMultiImage(object):
    """支持多模态图像（lidar_map/view_map）使用各自的归一化参数"""
    def __init__(self, 
                 img_keys=None, 
                 mean_lidar_map=None, 
                 std_lidar_map=None, 
                 mean_view_map=None, 
                 std_view_map=None, 
                 mean_single=None, 
                 std_single=None, 
                 to_rgb=True):
        # 初始化 lidar_map 的参数（3通道）
        self.mean_lidar = np.array(mean_lidar_map, dtype=np.float32) if mean_lidar_map is not None else None
        self.std_lidar = np.array(std_lidar_map, dtype=np.float32) if std_lidar_map is not None else None
        
        # 初始化 view_map 的参数（3通道）
        self.mean_view = np.array(mean_view_map, dtype=np.float32) if mean_view_map is not None else None
        self.std_view = np.array(std_view_map, dtype=np.float32) if std_view_map is not None else None
        
        # 单通道参数（如 z_map）
        if mean_single is None:
            # 默认使用 lidar 和 view 均值的平均
            self.mean_single = np.array([(np.mean(self.mean_lidar) + np.mean(self.mean_view)) / 2], dtype=np.float32)
        else:
            self.mean_single = np.array(mean_single, dtype=np.float32)
        if std_single is None:
            self.std_single = np.array([(np.mean(self.std_lidar) + np.mean(self.std_view)) / 2], dtype=np.float32)
        else:
            self.std_single = np.array(std_single, dtype=np.float32)
        
        self.to_rgb = to_rgb
        self.img_keys = img_keys

    def __call__(self, results):
        if self.img_keys is None:
            img_keys = list(results["imgs"].keys())
        else:
            img_keys = self.img_keys

        for img_key in img_keys:
            img = results["imgs"][img_key].copy()
            img = img.astype(np.float32) / 255.0  # 转换为 [0,1] 范围

            # 单通道图像扩维（z_map）
            if img.ndim == 2:
                img = img[..., np.newaxis]
                num_channels = 1
            else:
                num_channels = img.shape[-1]

            # 根据图像 key 选择对应的归一化参数
            if img_key == "lidar_map":
                assert self.mean_lidar is not None and self.std_lidar is not None, \
                    "lidar_map 的 mean/std 未配置"
                used_mean = self.mean_lidar
                used_std = self.std_lidar
                used_to_rgb = self.to_rgb 
            
            elif img_key == "view_map":
                assert self.mean_view is not None and self.std_view is not None, \
                    "view_map 的 mean/std 未配置"
                used_mean = self.mean_view
                used_std = self.std_view
                used_to_rgb = self.to_rgb  
            
            elif num_channels == 1:  # 单通道图像（如 z_map）
                used_mean = self.mean_single
                used_std = self.std_single
                used_to_rgb = False  # 单通道无需转 RGB
            
            else:
                raise ValueError(f"未支持的图像 key: {img_key}，请检查配置")

            # 执行归一化
            img_trans = mmcv.imnormalize(img, used_mean, used_std, used_to_rgb)
            results["imgs"][img_key] = img_trans
            results["img_metas"][img_key].update({
                'img_norm_cfg': dict(
                    mean=used_mean.tolist(),
                    std=used_std.tolist(),
                    to_rgb=used_to_rgb
                )
            })
        
        return results
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_keys={self.img_keys}, mean={self.mean}, std={self.std}, '
        repr_str += f'mean_single={self.mean_single}, std_single={self.std_single}, to_rgb={self.to_rgb})'
        return repr_str


# 以下 TrunkVMAAdaptor  PhotoMetricDistortionImage 
@PIPELINES.register_module()
class TrunkVMAAdaptor(object):
    def __init__(self):
        pass
    
    def __call__(self, results):
        # 1. 提取图像：去掉无意义的 transpose(0,1,2)，直接获取 (H,W,C) 格式的图
        imgs = [results["imgs"][img_key] for img_key in results["img_keys"]]
        
        # 2. 通道维拼接：得到 (1000,1000,7)
        imgs_concat = np.concatenate(imgs, axis=-1)
        
        # 3. 关键修复：调整维度顺序为「通道+高+宽」，并加批次维（适配PyTorch模型）
        # (H,W,C) → (C,H,W) → 加批次维 (1,C,H,W)
        imgs_tensor = to_tensor(imgs_concat).permute(2, 0, 1).unsqueeze(0)
        # 此时 imgs_tensor 形状为 (1,7,1000,1000)，符合模型输入要求
        
        # 4. 包装DC：保持后续标注处理逻辑不变
        results["img"] = DC(imgs_tensor, stack=True)
        # imgs = [results["imgs"][img_key].transpose(0, 1, 2) \
        #     for img_key in results["img_keys"]]
        # imgs = np.concatenate(imgs, axis=-1)  # 使用concatenate替代stack，axis=-1表示最后一个维度
        # results["img"] = DC(to_tensor(imgs), stack=True)

        ## map annots tensor
        gt_vecs_pts_loc = results['gt_vecs_pts_loc']
        gt_vecs_label = to_tensor(results['gt_vecs_label']).long()
        gt_vecs_attr = to_tensor(results['gt_vecs_attr']).long()
        if gt_vecs_attr.numel() != 0:
            gt_vecs_attr = gt_vecs_attr.permute(1, 0)
        
        results['gt_bboxes'] = DC(gt_vecs_pts_loc, cpu_only=True)
        results['gt_labels'] = DC(gt_vecs_label, cpu_only=False)
        results['gt_attrs'] = DC(gt_vecs_attr, cpu_only=False)
        
        # !: only take one img info for 'img_metas'
        one_img_key = results["img_keys"][0]
        img_metas = results["img_metas"][one_img_key] 
        results["img_metas"] = DC(img_metas, cpu_only=True)
        results["img_keys"] = DC(results["img_keys"], cpu_only=True)
        
        return results

@PIPELINES.register_module()
class PhotoMetricDistortionImage:
    """保持你原有的代码不变，无需修改"""
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        # 
        if not hasattr(self, 'img_keys'):
            self.img_keys = None
            
        if self.img_keys is None:
            img_keys = list(results["imgs"].keys())
        else:
            img_keys = self.img_keys

        for img_key in img_keys:
            img = results["imgs"][img_key].copy()
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # 修复2：单通道图跳过HSV转换（原代码会报错）
            num_channels = img.shape[-1] if img.ndim ==3 else 1
            if num_channels ==3:  # 只对3通道图做HSV转换
                # convert color from BGR to HSV
                img = mmcv.bgr2hsv(img)

                # random saturation
                if random.randint(2):
                    img[..., 1] *= random.uniform(self.saturation_lower,
                                                self.saturation_upper)

                # random hue
                if random.randint(2):
                    img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                    img[..., 0][img[..., 0] > 360] -= 360
                    img[..., 0][img[..., 0] < 0] += 360

                # convert color from HSV to BGR
                img = mmcv.hsv2bgr(img)

                # randomly swap channels
                if random.randint(2):
                    img = img[..., random.permutation(3)]

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # 修复3：键名错误（results['img']→results['imgs']）
            results['imgs'][img_key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str