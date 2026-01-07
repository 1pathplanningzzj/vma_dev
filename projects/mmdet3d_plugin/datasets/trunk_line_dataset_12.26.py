import os
import random
import os.path as osp
import copy
import logging
import json
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
import mmcv
from shapely.geometry import LineString, box, MultiLineString
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
# 获取当前文件路径
current_file = Path(__file__).resolve()
# 找到项目根目录（根据实际目录结构调整，这里假设根目录是当前文件的上3级目录）
project_root = current_file.parents[3]
# 将项目根目录添加到Python路径
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

#from pipelines import *

from projects.mmdet3d_plugin.datasets.sd_driving_line_dataset import \
    InstanceLines, VectorizedLocalMap, SD_Driving_Line_Dataset
from projects.mmdet3d_plugin.datasets.label_mapping import \
    LANELINE_LINTRYPE_MAPPING, LANELINE_SHAPE_MAPPING, CURB_LINETYPE_MAPPING, STOPLINE_LINETYPE_MAPPING
'''from sd_driving_line_dataset import \
    InstanceLines, VectorizedLocalMap, SD_Driving_Line_Dataset
from label_mapping import \
    LANELINE_LINTRYPE_MAPPING, LANELINE_SHAPE_MAPPING, CURB_LINETYPE_MAPPING, STOPLINE_LINETYPE_MAPPING'''


@DATASETS.register_module()
class TrunkLineDataset(SD_Driving_Line_Dataset):
    r'''
    DataLoader for sampling. Iterate the aerial images dataset
    '''
    def __init__(self, 
                 data_root,
                 imgs_dir,
                 annots_dir,
                 mask_dir, 
                 points_nums,
                 modality=None,
                 view_img_dir=None, #添加VIEW
                 z_map_dir=None, #添加z_map 高度信息
                 map_classes=None,
                 attrs_dict=None, 
                 eval_use_same_gt_sample_num_flag=False,
                 pipeline=None,
                 mode="valid", 
                 test_mode=False):
        assert mode in {"train", "test", "valid"}
        
        if modality is None:
            self.moality = dict(
                use_lidar_map=True,
                use_view_map=False,
                use_z_map=False,
            )
        else:
            self.moality = modality  
        
        self.MAPCLASSES = self.CLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.attrs_dict = attrs_dict
        
        self.data_root = data_root
        self.imgs_dir = imgs_dir
        self.annots_dir = annots_dir
        self.view_img_dir = view_img_dir #添加VIEW

        self.z_map_dir = z_map_dir #添加VIEW

        self.mask_dir = mask_dir # todo
        # load data
        self.annotation_dict = self.load_datadir()
        # # ! for debugging
        # self.annotation_dict = dict(list(self.annotation_dict.items())[4743: 4747])
        self.seq_len = len(self.annotation_dict)
        
        self.points_nums = points_nums
        
        self.eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag
        self.test_mode = test_mode
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if not test_mode:
            self._set_group_flag()
    
    def _preprocess_instances(self, instances):
        """map label to training format and filter out unwanted data 

        Args:
            instances (list): line instances
        """
        instances_new = []
        for instance in instances:
            instance_cat = instance['category']
            instance_data = instance['position']
            instance_attrs = instance['attributes']
            
             # !: filter out opposite lane_lines which are probably unclear in pcd
            if instance_attrs.get("direction") == "opposite_with_curb":
                continue
            if "line_type" in instance_attrs:
                line_type = "line_type"
            elif "curb_type" in instance_attrs:
                line_type = "curb_type"
            else:
                line_type="Type"
            ## map label & reconstruct attr
            if instance_cat == "lane_line":
                linetype = LANELINE_LINTRYPE_MAPPING.get(instance_attrs[line_type])
                shape = LANELINE_SHAPE_MAPPING.get(instance_attrs[line_type])
                if linetype is None:
                    # linetype = "other"
                    # shape = "other"
                    # logging.warning("Wrong label [{}] exists!".format(instance_attrs["line_type"]))
                    logging.warning("Skip wrong label [{}]!".format(instance_attrs[line_type]))#
                    continue
                    
                attr = {
                    "laneline_linetype": linetype,
                    "laneline_shape": shape,
                    "curb_linetype": "unknown",
                    "stopline_linetype": "unknown",
                }
            elif instance_cat == "curb":
                linetype = CURB_LINETYPE_MAPPING.get(instance_attrs[line_type])
                if linetype is None:
                    # linetype = "other"
                    # shape = "other"
                    # logging.warning("Wrong label [{}] exists!".format(instance_attrs["line_type"]))
                    logging.warning("Skip wrong label [{}]!".format(instance_attrs[line_type]))#
                    continue
                attr = {
                    "laneline_linetype": "no",
                    "laneline_shape": "unknown",
                    "curb_linetype": CURB_LINETYPE_MAPPING[instance_attrs[line_type]], # 
                    "stopline_linetype": "unknown",
                }
            elif instance_cat == "stop_line":
                attr = {
                    "laneline_linetype": "no",
                    "laneline_shape": "unknown",
                    "curb_linetype": "unknown",
                    "stopline_linetype": STOPLINE_LINETYPE_MAPPING[instance_attrs[line_type]],#
                }
            instances_new.append({"category": instance_cat, 
                                  "position": instance_data,
                                  "attributes": attr})
            
        return instances_new
        
        
    def load_datadir(self): #添加VIEW
        annotation_dict={}
        for bag_name in os.listdir(self.data_root):
            if ".bag" not in bag_name:
                print(bag_name)
                continue
            
            annots_dir_path = osp.join(self.data_root, bag_name, self.annots_dir)
            for annot_file in os.listdir(annots_dir_path):
                annot_path = osp.join(annots_dir_path, annot_file)
                annot = json.load(open(annot_path, "r"))
                
                image_name = "_".join([bag_name, annot["image_name"]])
                
                data_info = {
                    "instances": self._preprocess_instances(annot["instances"]),
                    "img_keys": [],
                    "img_paths": {},
                }
                
                if self.moality["use_lidar_map"]:
                    data_info["img_paths"]["lidar_map"] = osp.join(
                        self.data_root, bag_name, self.imgs_dir, annot["image_name"])
                    data_info["img_keys"].append("lidar_map")
                if self.moality["use_view_map"]:
                    data_info["img_paths"]["view_map"] = osp.join(
                        self.data_root, bag_name, self.view_img_dir, annot["image_name"])
                    data_info["img_keys"].append("view_map")
                if self.moality["use_z_map"]:
                    # to be implement
                    data_info["img_paths"]["z_map"] = osp.join(
                        self.data_root, bag_name, self.z_map_dir, annot["image_name"])
                    data_info["img_keys"].append("z_map")
                
                annotation_dict.update({image_name: data_info})
                
        return annotation_dict
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            # ! [temp]: avoid null gt data
            if data is None or len(data["gt_labels"].data) == 0:
                logging.warning(f"Data at index {idx} is invalid, randomly sampling another index.")
                idx = self._rand_another(idx)
                continue
            return data
    
    def _rand_another(self, idx):
        return random.randint(0, len(self) - 1)
    
    def prepare_train_data(self, idx):
        return self.load_train_data(list(self.annotation_dict.values())[idx])
    
    def prepare_test_data(self, idx):
        return self.load_test_data(list(self.annotation_dict.values())[idx])

    def load_train_data(self, data_info):
        data_info = copy.deepcopy(data_info)
        data_info = self.pipeline(data_info)
        return data_info
    
    def load_test_data(self, data_info):
        data_info = copy.deepcopy(data_info)
        data_info = self.pipeline(data_info)
        return data_info
    
    def _random_blur(self, img):
        """随机高斯模糊（降低图像清晰度，增强鲁棒性）"""
        #记录原格式信息（通道位置、维度），用于后续恢复
        original_shape = img.shape  # 记录原形状（如 (3, 1000, 1000)）
        is_channel_first = False    # 标记是否为“通道在前”格式
        
        # 2. 处理维度：确保转为 [H, W, C]
        if img.ndim == 3:
            # 情况1：通道在前（[C, H, W]，如模型常用格式）→ 转为 [H, W, C]
            if img.shape[0] in [1, 3]:  # 通道数为1（灰度）或3（彩色），判断为通道在前
                img = img.transpose(1, 2, 0)  # 转置维度：[C, H, W] → [H, W, C]
                is_channel_first = True
            # 情况2：通道在后（[H, W, C]）→ 无需处理
        elif img.ndim == 2:
            # 情况3：2维灰度图（[H, W]）→ 无需处理（OpenCV直接支持）
            pass
        else:
            # 情况4：维度异常（如含批次维度[B, C, H, W]）→ 报错并提示
            raise ValueError(f"不支持的图像维度：{img.ndim}，仅支持2维（灰度）或3维（彩色）图像")
        
        # 3. 随机高斯模糊（仅在概率触发时执行）
        if np.random.random() < 0.2:  # 20%概率触发模糊
            ksize = np.random.choice([3, 5])  # 模糊核大小（必须为奇数，OpenCV要求）
            # 调用OpenCV模糊函数（此时img已为 [H, W] 或 [H, W, C]）
            img = cv.GaussianBlur(img, (ksize, ksize), sigmaX=0)  # sigmaX=0：自动计算标准差
        
        # 4. 恢复原格式（确保后续模型输入兼容）
        if is_channel_first:
            # 通道在前格式：[H, W, C] → [C, H, W]
            img = img.transpose(2, 0, 1)
        # 验证形状是否与原始一致（避免维度错乱）
        assert img.shape == original_shape, \
            f"模糊后图像形状 {img.shape} 与原始形状 {original_shape} 不匹配"
        
        # 保持原数据类型（如float32）
        return img.astype(np.float32) 
    
    def _format_bbox(self, results, jsonfile_prefix=None):  #修改结果文件命名规则
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """

        print('Start to convert detection format...')
        results_dict = {}
        for result in results:
            pred_instances = []
            pred_scores = result['scores_3d'].numpy()
            pred_data = result['pts_3d'].numpy()
            pred_labels = result['labels_3d'].numpy()
            pred_attrs_label = torch.stack(result['attrs_3d']['attrs_preds']).transpose(1,0)
            for idx in range(len(pred_scores)):
                raw_points = pred_data[idx]  # [num_raw_points, 2]  
                
                # 2. 插值处理（参考 shift_fixed_num_sampled_points_v2 逻辑）
                if len(raw_points) >= 2:
                    line = LineString(raw_points)
                    distances = np.linspace(0, line.length, self.points_nums)  # 使用类中的 fixed_num
                    sampled_points = np.array([
                        list(line.interpolate(distance).coords)
                        for distance in distances
                    ]).reshape(-1, 2)  # [fixed_num, 2]
                else:
                    # 处理点数不足的情况（如预测失败）
                    sampled_points = np.zeros((self.points_nums, 2))
                pred_instances.append({'class':pred_labels[idx], \
                                        'data':pred_data[idx], \
                                        'res_data':sampled_points.tolist(),\
                                        'attrs':pred_attrs_label[idx].tolist(), \
                                        'confidence_level':pred_scores[idx]})
            image_path = result['img_metas']['filename']
            image_name = image_path.split('/')[-4]+'/'+image_path.split('/')[-1]  #修改命名
            results_dict.update({image_name:{'image_path':image_path, 'pred_instances':pred_instances}})
        res_path = osp.join(jsonfile_prefix, 'results_fuse175_test316.json')
        print('Results writes to', res_path)
        
        mmcv.dump(results_dict, res_path)
        return res_path
    
    def _evaluate_single(self,
                     result_path,
                     logger=None,
                     metric='chamfer',
                     result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol."""
        from projects.mmdet3d_plugin.datasets.map_utils.trunk_line_evaluate import eval_map, format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        
        print('Formating results & gts by classes')
        with open(result_path,'r') as f:
            pred_results = json.load(f)
        gen_results, annotations = self.arrange_results_and_annotations(pred_results, self.annotation_dict)
        detail = dict()  # 存储所有评估指标的字典
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                    gen_results,
                                                    annotations,
                                                    cls_names=self.MAPCLASSES,
                                                    num_pred_pts_per_instance=self.points_nums,
                                                    eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag)
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        
        for metric in metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)

            if metric == 'chamfer':
                thresholds = [3, 6, 15]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))
            
            # 新增：存储每个阈值下的属性子组指标
            attr_subgroup_aps = {}  # 结构: {阈值: {类别: {属性: {取值: AP}}}}
            
            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(  # cls_ap包含每个类别的attr_subgroups
                                gen_results,
                                annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                attrs_dict=self.attrs_dict,
                                logger=logger,
                                num_pred_pts_per_instance=self.points_nums,
                                metric=metric)
                
                # 记录类别级AP
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']
                
                # 新增：记录属性子组AP（按当前阈值）
                attr_subgroup_aps[thr] = {}
                for j, clsname in enumerate(self.MAPCLASSES):
                    attr_subgroup_aps[thr][clsname] = cls_ap[j].get('attr_subgroups', {})
            
            for i, name in enumerate(self.MAPCLASSES):
                print(f'{name}: {cls_aps.mean(0)[i]:.3f}')
                detail[f'SD_Line_Map_{metric}/{name}_AP'] = cls_aps.mean(0)[i]
            print(f'map: {cls_aps.mean(0).mean():.3f}')
            detail[f'SD_Line_Map_{metric}/mAP'] = cls_aps.mean(0).mean()
            
            # 原有：存储每个阈值的类别级AP
            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer' or (metric == 'iou' and thr in [0.5, 0.75]):
                        detail[f'SD_Line_Map_{metric}/{name}_AP_thr_{thr}'] = cls_aps[j][i]
            
            for thr in thresholds:
                if metric == 'iou' and thr not in [0.5, 0.75]:
                    continue
                
                for cls_idx, clsname in enumerate(self.MAPCLASSES):
                    subgroups = attr_subgroup_aps[thr].get(clsname, {})
                    for attr_name, value_results in subgroups.items():
                        for value, sub_result in value_results.items():
                            key = f'SD_Line_Map_{metric}/{clsname}_{attr_name}_{value}_AP_thr_{thr}'
                            detail[key] = sub_result['ap']
                    
                    # 新增：存储属性子组的平均AP（每个属性的所有取值平均）
                    for attr_name in subgroups.keys():
                        values = subgroups[attr_name].values()
                        if not values:
                            continue
                        avg_ap = np.mean([v['ap'] for v in values if not np.isnan(v['ap'])])
                        key_avg = f'SD_Line_Map_{metric}/{clsname}_{attr_name}_avg_AP_thr_{thr}'
                        detail[key_avg] = avg_ap

        return detail


    def arrange_results_and_annotations(self, results, annotations):
        # this function is to arrange one big class together
        all_image_arranged_pred_results = []
        all_image_arranged_gt_annotations = []
        for pred_result, gt_annotation in zip(results.values(), annotations.values()):
            single_image_pred_results = []
            single_image_gt_annotations = []
            pred_instances = pred_result['pred_instances']
            for instance in pred_instances:
                instance_data = [[int(x[0]), int(x[1])] for x in instance['data']]
                single_image_pred_results.append({'pts':instance_data,'type':instance['class'], 'attrs':instance['attrs'], 'confidence_level':instance['confidence_level']})
            gt_instances = gt_annotation['instances']
            for instance in gt_instances:
                instance_cat = instance['category']
                instance_data = instance['position']
                instance_attrs = instance['attributes']
                
                if instance_cat == 'lane_line':
                    instance_attrs = [
                        self.attrs_dict["laneline_linetype"].index(instance_attrs["laneline_linetype"]),
                        self.attrs_dict["laneline_shape"].index(instance_attrs["laneline_shape"]),
                    ]
                elif instance_cat == 'curb':
                    instance_attrs = [self.attrs_dict['curb_linetype'].index(instance_attrs['curb_linetype'])]
                elif instance_cat == 'stop_line':
                    instance_attrs = [self.attrs_dict['stopline_linetype'].index(instance_attrs['stopline_linetype'])]
                    
                single_image_gt_annotations.append({'pts':instance_data,'type':self.MAPCLASSES.index(instance_cat), 'attrs':instance_attrs})
            all_image_arranged_pred_results.append(single_image_pred_results)
            all_image_arranged_gt_annotations.append(single_image_gt_annotations)
        return all_image_arranged_pred_results, all_image_arranged_gt_annotations
    
    def show_result_with_original_image(self,
                                        pred_instances,
                                        out_dir,
                                        image_path): #修改可视化文件存储路径
        path_parts = image_path.split('/')
        bag_folder = None
        for part in path_parts:
            if '.bag' in part:
                bag_folder = part
                break
        # 构建输出子目录：out_dir/bag_folder
        if bag_folder:
            out_subdir = os.path.join(out_dir, bag_folder)
            # 创建子目录（若不存在）
            os.makedirs(out_subdir, exist_ok=True)
        else:
            # 若未找到bag文件夹，默认用out_dir
            out_subdir = out_dir
        image_name = image_path.split('/')[-1]
        image = cv.imread(image_path)
        color = {0:(0, 0, 0), 1:(0, 0, 255), 2:((0, 97,255))}
        attrs_map=[
            [ "single_solid", "single_dash", "double_solid", "double_dash", "thick_dash",
                        "other", "unknown", "no"],
            ["road_boundary", "cone_boundary", "other", "unknown"],
            ["normal", "other", "unknown"]
        ]
        
        # drew_image_path = out_dir + '/' + image_name
        # cv.imwrite(drew_image_path, image)
        for pred_instance in pred_instances:
            # import pdb;pdb.set_trace()
            sub_pred_points = pred_instance['data']
            pred_class = pred_instance['class']
            pred_attr = pred_instance['attrs']
            
            # import pdb;pdb.set_trace()
            color_fill = color[pred_class]   
            sub_pred_points = [tuple([int(x[0]), int(x[1])]) for x in sub_pred_points]
            if pred_class == 2:
                for i in range(1, len(sub_pred_points)):
                    cv.line(image, sub_pred_points[i-1], sub_pred_points[i], color_fill, 4)
            else:
                for i in range(len(sub_pred_points)):
                    if i == 0:
                        cv.circle(image, sub_pred_points[i], 10, (0,97,255), -1)
                        continue
                    elif i == len(sub_pred_points)-1:
                        cv.line(image, sub_pred_points[i-1], sub_pred_points[i], color_fill, 4)
                        cv.circle(image, sub_pred_points[i], 8, (0,255,255), -1)
                    else:
                        cv.line(image, sub_pred_points[i-1], sub_pred_points[i], color_fill, 4)
                        cv.circle(image, sub_pred_points[i], 5, (0, 255, 0), -1)
                if pred_class==0:
                    text=attrs_map[pred_class][pred_attr[pred_class]]
                else:
                    text=attrs_map[pred_class][pred_attr[pred_class+1]]
                ypos=sub_pred_points[0][1]-15
                if sub_pred_points[0][1]<10:
                    ypos=sub_pred_points[0][1]+15
                cv.putText(
                    image,
                    text,
                    (sub_pred_points[0][0]-5, ypos),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,  # 字体大小
                    (0, 0, 0),  # 黑色背景
                    3,  # 背景线宽
                    cv.LINE_AA
                )
                cv.putText(
                    image,
                    text,
                    (sub_pred_points[0][0]-5, ypos),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,  # 字体大小
                    (255, 255, 255),  # 白色字体
                    1,  # 字体线宽
                    cv.LINE_AA
                )

        original_image = cv.imread(image_path)
        drew_image = cv.copyMakeBorder(image, 0, 0, 50, 0, cv.BORDER_CONSTANT, value=(128,128,128))
        original_image = cv.imread(image_path)
        concat_image = cv.hconcat([original_image, drew_image])
        drew_image_path = out_subdir + '/' + image_name
        cv.imwrite(drew_image_path, concat_image)


if __name__ == "__main__":
    map_classes = ['lane_line', 'curb', 'stop_line']
    img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=False)
    pipline = [
    dict(type='LoadImageFromFiles', to_float32=True),
    dict(type='NormalizeImage', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['img_shape', 'filename'],
    )]
    attrs_dict = dict(
    #     laneline_linetype=[ "single_solid", "single_dash", "double_solid", "double_dash", "thick_dash",                                    # "left_wait_line",  "double_soild", "double_right_soild", "double_left_soild", 
    laneline_linetype=[ "single_solid", "single_dash", "double_solid", "double_dash", "thick_dash",
                        "other", "unknown", "no"],
    laneline_shape=["normal", "fishbone", "unknown"],
    # laneline_direction=["same", "opposite_with_curb", "opposite_without_curb", "unknown"],
    curb_linetype=["road_boundary", "cone_boundary", "other", "unknown"],
    stopline_linetype=["normal", "other", "unknown"],
    )
    data=TrunkLineDataset(data_root="/homes/zhangzijian/vma-dev/data/train_nas316/train_nas316",
                          imgs_dir="cropped_data/images",
                          annots_dir="cropped_data/annots",
                          mask_dir=None,
                          points_nums=50,
                          view_img_dir="cropped_data/view_maps",
                          z_map_dir = "cropped_data/z_maps_vis",

                          map_classes=map_classes,
                          attrs_dict=attrs_dict,
                          eval_use_same_gt_sample_num_flag=True,
                          pipeline=pipline
                          )
    print(data[10])