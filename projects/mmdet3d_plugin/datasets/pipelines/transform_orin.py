import numpy as np
from numpy import random
import cv2
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

@PIPELINES.register_module()
class NormalizeMultiImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, img_keys=None, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        
        self.img_keys = img_keys

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        if self.img_keys is None:
            img_keys = list(results["imgs"].keys())
        else:
            img_keys = self.img_keys

        for img_key in img_keys:
            img = results["imgs"][img_key].copy()
            img /= 255
            img_trans = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            # import pdb; pdb.set_trace()
            results["imgs"][img_key] = img_trans
            results["img_metas"][img_key].update({
                'img_norm_cfg': dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
            })
        
        # cv2.imwrite("aug_norm.jpg", results["imgs"][img_key])
        # import pdb; pdb.set_trace()
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_keys={self.img_keys}, mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
    
@PIPELINES.register_module()
class TrunkVMAAdaptor(object):
    def __init__(self):
        pass
    
    def __call__(self, results):
        imgs = [results["imgs"][img_key].transpose(2, 0, 1) \
            for img_key in results["img_keys"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        results["img"] = DC(to_tensor(imgs), stack=False)
        
        ## map annots tensor
        gt_vecs_pts_loc = results['gt_vecs_pts_loc']
        gt_vecs_label = to_tensor(results['gt_vecs_label']).long()
        gt_vecs_attr = to_tensor(results['gt_vecs_attr']).long()
        if gt_vecs_attr.numel() != 0:
            gt_vecs_attr = gt_vecs_attr.permute(1, 0)
        
        # if isinstance(results['gt_vecs_pts_loc'], InstanceLines):
        #     gt_vecs_pts_loc = results['gt_vecs_pts_loc']
        # else:
        #     gt_vecs_pts_loc = to_tensor(results['gt_vecs_pts_loc'])
        #     try:
        #         gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
        #     except:
        #         gt_vecs_pts_loc = gt_vecs_pts_loc
        
        results['gt_bboxes'] = DC(gt_vecs_pts_loc, cpu_only=True)
        results['gt_labels'] = DC(gt_vecs_label, cpu_only=False)
        results['gt_attrs'] = DC(gt_vecs_attr, cpu_only=False)
        
        # !: only take one img info for 'img_metas'
        one_img_key = results["img_keys"][0]
        img_metas = results["img_metas"][one_img_key] 
        results["img_metas"] = DC(img_metas, cpu_only=True)
        results["img_keys"] = DC(results["img_keys"], cpu_only=True)
        
        return results

# todo
@PIPELINES.register_module()
class PhotoMetricDistortionImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """
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
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
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

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            results['img'][img_key] = img
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