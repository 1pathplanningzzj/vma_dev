from typing import Union
import torch
import numpy as np
from numpy import random
import cv2
import copy
import mmcv
from shapely.geometry import LineString, box

from mmdet.datasets.builder import PIPELINES

# todo(rui): consider polygon cropping
def patch_crop_line(line: np.array, patch: Union[list, tuple]):
    """ Crop line to fit into the cropping patch
    
    Args:
        line: Array [N, 2]
        patch: (x_min, y_min, x_max, y_max)
    """
    line_shapley = LineString(line)
    patch = box(*patch)
    new_line = line_shapley.intersection(patch)
    
    line_list = []
    if not new_line.is_empty:
        if new_line.geom_type == 'MultiLineString':
            for single_line in new_line.geoms:
                if single_line.is_empty:
                    continue
                line_list.append(np.array(single_line.coords))
        else:
            line_list.append(np.array(new_line.coords))
            
    return line_list
    
@PIPELINES.register_module()
class PointEnhanceLowIntensityTarget(object):
    '''
    date: 2025.10.15
    function: 新增点云数据增强操作
    未使用 在生成vma 训练数据时使用 
    此 功能已经数据预处理的时候实现 
    '''
    def __init__(
        self,
        img_keys=["lidar_map"],  # 恢复默认值，避免为空
        colormap=2,  # 用数值2替代cv2.COLORMAP_JET，避免配置文件依赖cv2
        weak_thr=0.2,
        enhance_method="log",
        log_gain=15.0,
        power_gamma=0.4,
        intensity_weight=1.8
    ):
        assert enhance_method in ["log", "power"], "增强方法仅支持 'log' 或 'power'"
        assert 0 < weak_thr < 1, "弱强度阈值需在0-1之间"
        
        self.img_keys = img_keys
        self.colormap = colormap
        self.weak_thr = weak_thr
        self.enhance_method = enhance_method
        self.log_gain = log_gain
        self.power_gamma = power_gamma
        self.intensity_weight = intensity_weight
        
        # 预生成颜色映射表（用于从彩色值反推强度）
        self.lut = self._create_colormap_lut()

    def _create_colormap_lut(self):
        """生成颜色映射表：强度值(0-255)→彩色值(BGR)"""
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            # 确保cv2正确导入且颜色映射生效
            lut[i] = cv2.applyColorMap(np.uint8([i]), self.colormap)[0][0]
        return lut

    def _color_to_intensity(self, color_img):
        """从彩色图反推原始强度值（0-255）→ 优化效率"""
        h, w = color_img.shape[:2]
        # 向量化操作替代双重循环，提升速度
        color_flat = color_img.reshape(-1, 3)
        dist = np.sum((self.lut[:, np.newaxis] - color_flat) **2, axis=2)
        intensity_flat = np.argmin(dist, axis=0)
        return intensity_flat.reshape(h, w)

    def _normalize(self, arr):
        """归一化到0-1范围"""
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def _enhance_weak(self, intensity):
        """增强弱强度区域"""
        intensity_norm = self._normalize(intensity)
        mask_weak = intensity_norm < self.weak_thr
        intensity_enh = intensity_norm.copy()
        
        if self.enhance_method == "log":
            intensity_enh[mask_weak] = np.log1p(self.log_gain * intensity_enh[mask_weak])
        else:
            intensity_enh[mask_weak] = np.power(intensity_enh[mask_weak], self.power_gamma)
        
        # 恢复到0-255范围并应用权重
        intensity_enh = self._normalize(intensity_enh)
        intensity_enh = (intensity_enh * 255 * self.intensity_weight).clip(0, 255).astype(np.uint8)
        return intensity_enh

    def __call__(self, results):
        # 确保img_keys有效
        if not self.img_keys:
            self.img_keys = list(results["imgs"].keys())
            
        for img_key in self.img_keys:
            # 1. 读取三通道彩色图
            color_img = results["imgs"][img_key].copy()
            if color_img.ndim != 3 or color_img.shape[2] != 3:
                raise ValueError(f"{img_key} 不是三通道彩色图")
            
            # 2. 从彩色图反推强度值
            intensity = self._color_to_intensity(color_img)
            
            # 3. 增强弱强度
            intensity_enh = self._enhance_weak(intensity)
            
            # 4. 用增强后的强度重建彩色图
            enhanced_color = cv2.applyColorMap(intensity_enh, self.colormap)
            
            # 5. 转换为float32格式（与后续pipeline兼容）
            enhanced_color = enhanced_color.astype(np.float32) / 255.0  # 归一化到0-1
            
            # 6. 更新结果
            results["imgs"][img_key] = enhanced_color
            results["img_metas"][img_key].update({
                "weak_intensity_enhanced": True,
                "enhance_method": self.enhance_method,
                "dtype": "float32"
            })
        
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(img_keys={self.img_keys}, method={self.enhance_method})"

@PIPELINES.register_module()
class CropResizeFlipImage(object):
    """
    Apply in order (a) crop (b) resize (c) flip to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        img_keys (List[str]): image keys to apply transformation. If None,
        apply for all images.
        resize_shape (tuple): final resize shape (w, h).
        crop_center (tuple): Center point (w, h) of the
            cropping in the source image. If None, the center of the
            image will be used.
        crop_ratio (List[tuple] | tuple): Crop ratio 
            [(rw_min, rw_max), (rh_min, rh_max)] or (r_min, r_max) which
            shares the same params for both w & h.
        keep_crop_region (bool): if true, the cropped region will be kept;
            otherwise, it will be removed. 
        rand_hflip (bool): Whether flip horizontally or not.
        flip_prob (float): The probability for flip and
            should be in range 0 to 1.
    """
    
    def __init__(
        self,
        prob=0.5,
        img_keys=None,
        resize_shape=None,
        crop_center=None,
        crop_ratio=(1, 1),
        keep_crop_region=True,
        rand_hflip=False,
        flip_prob=0.5,
    ):
        assert 0 <= prob <= 1.0, 'The probability of shear should be in ' \
            f'range [0,1]. got {prob}.'
        if isinstance(resize_shape, tuple):
            assert len(resize_shape) == 2, 'resize_shape as tuple must ' \
                f'have 2 elements. got {len(resize_shape)}.'
        else:
            raise ValueError(
                'resize_shape must be float or tuple with 2 elements.')
        if isinstance(crop_ratio, tuple):
            crop_ratio = [crop_ratio] * 2
        elif isinstance(crop_ratio, tuple):
            assert len(crop_ratio) == 2, 'crop_ratio as list must ' \
                f'have 2 elements. got {len(crop_ratio)}.'
            assert len(crop_ratio[0]) == 2, 'the element in crop ratio must' \
                f'have 2 elements. got {len(crop_ratio[0])}.'
        else:
            raise ValueError(
                'crop_ratio must be float or tuple with 2 elements.')
            
        self.prob = prob
        self.img_keys = img_keys
        self.resize_shape = resize_shape
        self.crop_center = crop_center
        self.crop_ratio = crop_ratio
        self.keep_crop_region = keep_crop_region
        self.rand_hflip = rand_hflip
        self.flip_prob = flip_prob
        
    def _transform_img(self, results, crop_patch, hflip):
        
        if self.img_keys is None:
            img_keys = list(results["imgs"].keys())
        else:
            img_keys = self.img_keys

        for img_key in img_keys:
            img = results["imgs"][img_key].copy()
            
            x1, y1, x2, y2 = crop_patch
            cropped = img[y1:y2, x1:x2]
            if self.keep_crop_region:
                new_img = np.zeros_like(img)
                new_img[y1:y2, x1:x2] = cropped
                img = new_img
            else:
                img = cropped

            h, w = img.shape[:2]
            if not (w, h) == self.resize_shape: # ! do not resize if shares the same size
                img = cv2.resize(img, self.resize_shape)
            
            if hflip:
                img = cv2.flip(img, 1)
                
            results["imgs"][img_key] = img
            results["img_metas"][img_key].update({
                "img_shape": img.shape[:2],
                "crop": crop_patch,
                "keep_crop_region": self.keep_crop_region,
                "hflip": hflip,
            })
            
    def _transform_instances(self, results, patch, trans_matrix):
        x1, y1, x2, y2 = patch
        patch = (x1, y1, x2 - 1, y2 - 1)
        
        instances_trans = []
        instances = results["instances"]
        for inst in instances:
            pos = np.array(inst["position"])
            # crop first
            poses_copped = patch_crop_line(pos, patch) 
            
            for pos in poses_copped:
                pos_homo = np.hstack([pos, np.ones_like(pos[:, :1])])
                # then apply resize and flip 
                pos_trans = trans_matrix @ pos_homo.T
                pos_trans = pos_trans[:2].T
                
                inst_new = copy.deepcopy(inst)
                inst_new["position"] = pos_trans
                instances_trans.append(inst_new)
        
        results["instances"] = instances_trans
    
    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        hflip = self.rand_hflip and np.random.rand() < self.flip_prob
        
        h, w = list(results["img_metas"].values())[0]["img_shape"]
        
        rw = random.uniform(*self.crop_ratio[0])
        rh = random.uniform(*self.crop_ratio[1])
        crop_w, crop_h = int(w * rw), int(h * rh)
        cx, cy = self.crop_center if self.crop_center is not None \
            else (w // 2, h // 2)
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        crop_patch= (x1, y1, x2, y2)

        crop_trans = np.eye(3)
        if not self.keep_crop_region:
            crop_trans[:2, 2] = np.array([-x1, -y1])
        resize_trans = np.eye(3)
        if not self.keep_crop_region:
            resize_trans[0, 0] = self.resize_shape[0] / (x2 - x1)
            resize_trans[1, 1] = self.resize_shape[1] / (y2 - y1)
        else:
            resize_trans[0, 0] = self.resize_shape[0] / w
            resize_trans[1, 1] = self.resize_shape[1] / h
        hflip_trans = np.eye(3)
        if hflip:
            hflip_trans[0, 0] = -1
            hflip_trans[0, 2] = self.resize_shape[0] - 1
        trans_matrix = hflip_trans @ resize_trans @ crop_trans
        
        self._transform_img(results, crop_patch, hflip)
        self._transform_instances(results, crop_patch, trans_matrix)
        
        return results
        
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'(img_keys={self.img_keys}, '
        repr_str += f'(resize_shape={self.resize_shape}, '
        repr_str += f'(crop_center={self.crop_center}, '
        repr_str += f'(crop_ratio={self.crop_ratio}, '
        repr_str += f'(keep_crop_region={self.keep_crop_region}, '
        repr_str += f'(rand_hflip={self.rand_hflip}, '
        repr_str += f"flip_prob='{self.flip_prob}')"
        return repr_str
            

@PIPELINES.register_module()
class AffineTransform(object):
    """
    Apply Affine Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        img_keys (List[str]): image keys to apply transformation. If None,
        apply for all images.
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``mmcv.imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        max_rotate_angle (int | float): The maximum angles for rotate
            transformation.
        max_translate_offset (int | float | tuple): The maximum translation
            ratio regarding image's w & h. If float, the same value will be 
            used for both width and height. If tuple, it should be 2 elements.
        scale_range (tuple): (scale_ratio_min, scale_ratio_max)
    """
    
    def __init__(
        self,
        img_keys=None,
        center=None,
        img_fill_val=0,
        prob=0.5,
        max_rotate_angle=30,
        max_translation_ratio=0.1,
        scale_range=(1, 1),
    ):
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must ' \
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), 'all ' \
            'elements of img_fill_val should between range [0,255].' \
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability of shear should be in ' \
            f'range [0,1]. got {prob}.'
        if isinstance(max_translation_ratio, (float, int)):
            max_translation_ratio = tuple([float(max_translation_ratio)] * 2)
        elif isinstance(max_translation_ratio, tuple):
            assert len(max_translation_ratio) == 2, 'max_translation_ratio as tuple must ' \
                f'have 2 elements. got {len(max_translation_ratio)}.'
            max_translation_ratio = tuple([float(val) for val in max_translation_ratio])
        else:
            raise ValueError(
                'max_translation_ratio must be float or tuple with 2 elements.')
        if isinstance(scale_range, tuple):
            assert len(scale_range) == 2, 'scale_range as tuple must ' \
                f'have 2 elements. got {len(scale_range)}.'
            scale_range = tuple([float(val) for val in scale_range])
        else:
            raise ValueError(
                'scale_range must be float or tuple with 2 elements.')
        
        self.img_keys = img_keys
        self.center = center
        self.img_fill_val = img_fill_val
        self.prob = prob
        self.max_rotate_angle = max_rotate_angle
        self.max_translation_ratio = max_translation_ratio
        self.scale_range = scale_range
        
    def _transform_img(self, results, trans_matrix):
        if self.img_keys is None:
            img_keys = list(results["imgs"].keys())
        else:
            img_keys = self.img_keys

        for img_key in img_keys:
            img = results["imgs"][img_key].copy()
            h, w = img.shape[:2]
            img_trans = cv2.warpAffine(
                img, 
                trans_matrix, 
                (w, h), 
                flags=cv2.INTER_LINEAR, 
                borderValue=self.img_fill_val,
            )
            results["imgs"][img_key] = img_trans
            results["img_metas"][img_key].update({
                "affine_transform": trans_matrix
            })
    
    def _transform_instances(self, results, trans_matrix, patch):
        instances_trans = []
        instances = results["instances"]
        for inst in instances:
            pos = np.array(inst["position"])
            pos_homo = np.hstack([pos, np.ones_like(pos[:, :1])])
            pos_trans = trans_matrix @ pos_homo.T
            pos_trans = pos_trans[:2].T
            
            poses_copped = patch_crop_line(pos_trans, patch) # crop into patch
            
            for pos_cropped in poses_copped:
                inst_new = copy.deepcopy(inst)
                inst_new["position"] = pos_cropped
                instances_trans.append(inst_new)
        
        results["instances"] = instances_trans
            
    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        
        h, w = list(results["img_metas"].values())[0]["img_shape"]
        patch = (0, 0, w, h)
        if self.center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        else:
            center = self.center
        angle = np.random.uniform(-self.max_rotate_angle, self.max_rotate_angle)
        translation = (
            np.random.uniform(-self.max_translation_ratio[0]*w, self.max_translation_ratio[0]*w),
            np.random.uniform(-self.max_translation_ratio[1]*h, self.max_translation_ratio[1]*h),
        )
        scale = np.random.uniform(*self.scale_range)
        
        trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        trans_matrix[0, 2] += translation[0]
        trans_matrix[1, 2] += translation[1]
        
        self._transform_img(results, trans_matrix)
        self._transform_instances(results, trans_matrix, patch)
        
        # # !: [vis]
        # img = results["imgs"]["lidar_map"]
        # insts = results["instances"]
        # for inst in insts:
        #     pts = inst["position"].astype(np.int32)
        #     cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=2)
        #     for (x, y) in pts:
        #         cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        # cv2.imwrite("aug_rot.jpg", img)
        # import pdb; pdb.set_trace()
        return results
        
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'(img_keys={self.img_keys}, '
        repr_str += f'(center={self.center}, '
        repr_str += f'(img_fill_val={self.img_fill_val}, '
        repr_str += f'(max_rotate_angle={self.max_rotate_angle}, '
        repr_str += f'(max_translation_ratio={self.max_translation_ratio}, '
        repr_str += f"scale_range='{self.scale_range}')"
        return repr_str  