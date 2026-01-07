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
    LANELINE_LINETYPE_MAPPING, \
    LANELINE_FUNCTION_MAPPING, \
    LANELINE_COLOR_MAPPING, \
    CURB_LINETYPE_MAPPING, \
    STOPLINE_LINETYPE_MAPPING

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
                test_mode=False,
                # 新增：接收动态采样配置
                dynamic_sample_config=None):
        assert mode in {"train", "test", "valid"}
        
        if modality is None:
            self.moality = dict(
                use_lidar_map=True,
                use_view_map=False,
                use_z_map=False,
            )
        else:
            self.moality = modality  
        
        # 新增：初始化动态采样参数（默认值兜底）
        self.dynamic_sample_config = dynamic_sample_config or dict(
            sample_density=2.0,
            min_sample_points=5,
            max_sample_points=25,
            bev_res=0.05
        )
        self.sample_density = self.dynamic_sample_config['sample_density']
        self.min_sample_points = self.dynamic_sample_config['min_sample_points']
        self.max_sample_points = self.dynamic_sample_config['max_sample_points']
        self.bev_res = self.dynamic_sample_config['bev_res']
        
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
        instances_new = []

        for instance in instances:
            instance_cat = instance['category']
            instance_data = instance['position']
            instance_attrs = instance['attributes']

            # 过滤反向车道线
            if instance_attrs.get("direction") == "opposite_with_curb":
                continue

            # ---------------- lane_line ----------------
            if instance_cat == "lane_line":
                # 原始字段
                line_type = instance_attrs.get("line_type")
                line_func = instance_attrs.get("line_function")
                line_color = instance_attrs.get("line_color", "other")

                # 映射
                linetype = LANELINE_LINETYPE_MAPPING.get(line_type)
                function = LANELINE_FUNCTION_MAPPING.get(line_func)
                color = LANELINE_COLOR_MAPPING.get(line_color, "other")

                # lane_line → stop_line 下沉
                if function == "stop_line":
                    instance_cat = "stop_line"
                    attr = {
                        "laneline_linetype": "no",
                        "laneline_function": "no",
                        "laneline_color": "no",
                        "curb_linetype": "unknown",
                        "stopline_linetype": "normal",
                    }
                else:
                    if linetype is None or function is None or color is None:
                        logging.warning(
                            "[Skip lane_line after mapping] "
                            f"line_type={line_type}->{linetype}, "
                            f"line_function={line_func}->{function}, "
                            f"line_color={line_color}->{color}"
                        )
                        continue

                    attr = {
                        "laneline_linetype": linetype,
                        "laneline_function": function,
                        "laneline_color": color,
                        "curb_linetype": "unknown",
                        "stopline_linetype": "unknown",
                    }


            # ---------------- curb ----------------
            elif instance_cat == "curb":
                curb_type = instance_attrs.get("curb_type")
                linetype = CURB_LINETYPE_MAPPING.get(curb_type)
                if linetype is None:
                    logging.warning(f"Skip wrong curb label: {instance_attrs}")
                    continue

                attr = {
                    "laneline_linetype": "no",
                    "laneline_function": "no",
                    "laneline_color": "no",
                    "curb_linetype": linetype,
                    "stopline_linetype": "unknown",
                }

            # ---------------- stop_line ----------------
            elif instance_cat == "stop_line":
                attr = {
                    "laneline_linetype": "no",
                    "laneline_function": "no",
                    "laneline_color": "no",
                    "curb_linetype": "unknown",
                    "stopline_linetype": STOPLINE_LINETYPE_MAPPING.get(
                        instance_attrs.get("stopline_type", "normal"),
                        "other"
                    ),
                }
            else:
                continue

            instances_new.append({
                "category": instance_cat,
                "position": instance_data,
                "attributes": attr
            })

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
                # ========== 关键修复：添加异常捕获 ==========
                try:
                    # 用with语句安全读取，指定UTF-8编码
                    with open(annot_path, "r", encoding="utf-8") as f:
                        annot = json.load(f)
                except json.JSONDecodeError as e:
                    # 打印损坏文件路径+错误位置，方便后续手动修复
                    print(f"【跳过损坏JSON】文件：{annot_path}，错误位置：第{e.lineno}行第{e.colno}列")
                    continue  # 跳过该文件，继续执行
                except Exception as e:
                    # 捕获其他读取错误（如文件不存在、权限问题）
                    print(f"【跳过读取失败文件】文件：{annot_path}，错误：{str(e)}")
                    continue
                # ==============================================
                
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
                raw_points = pred_data[idx]  # [num_raw_points, 2] (像素单位)  
                
                # 核心：动态计算采样点数
                if len(raw_points) >= 2:
                    # 1. 转换为LineString，计算像素长度
                    line = LineString(raw_points)
                    pixel_length = line.length
                    # 2. 像素转米（关键：乘以bev_res）
                    line_length = pixel_length * self.bev_res
                    # 3. 动态计算采样点数 = 长度 / 采样密度（取整）
                    sample_points_num = int(round(line_length / self.sample_density))
                    # 4. 兜底：限制点数范围
                    sample_points_num = max(self.min_sample_points, min(sample_points_num, self.max_sample_points))
                    # 5. 按动态点数插值
                    distances = np.linspace(0, line.length, sample_points_num)
                    sampled_points = np.array([
                        list(line.interpolate(distance).coords)
                        for distance in distances
                    ]).reshape(-1, 2)  # [动态点数, 2]
                else:
                    # 点数不足时，用最小点数填充全0
                    sampled_points = np.zeros((self.min_sample_points, 2))
                
                pred_instances.append({
                    'class':pred_labels[idx], 
                    'data':pred_data[idx],  # 模型原始输出的25个点（像素）
                    'res_data':sampled_points.tolist(),# 动态插值后的点（像素）
                    'attrs':pred_attrs_label[idx].tolist(), 
                    'confidence_level':pred_scores[idx],
                    # 新增：记录长度和采样点数（方便分析）
                    'pixel_length': pixel_length if len(raw_points)>=2 else 0.0,
                    'line_length_m': line_length if len(raw_points)>=2 else 0.0,
                    'sample_points_num': len(sampled_points)
                })
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
            
            # 原有：存储类别级平均AP
            for i, name in enumerate(self.MAPCLASSES):
                print(f'{name}: {cls_aps.mean(0)[i]:.3f}')
                detail[f'SD_Line_Map_{metric}/{name}_AP'] = cls_aps.mean(0)[i]
            print(f'map: {cls_aps.mean(0).mean():.3f}')
            detail[f'SD_Line_Map_{metric}/mAP'] = cls_aps.mean(0).mean()
            
            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer' or (metric == 'iou' and thr in [0.5, 0.75]):
                        detail[f'SD_Line_Map_{metric}/{name}_AP_thr_{thr}'] = cls_aps[j][i]
            
            # 新增：存储属性子组的AP（按阈值、类别、属性、取值）
            for thr in thresholds:
                # 过滤不需要记录的阈值（如iou仅保留0.5和0.75）
                if metric == 'iou' and thr not in [0.5, 0.75]:
                    continue
                
                for cls_idx, clsname in enumerate(self.MAPCLASSES):
                    subgroups = attr_subgroup_aps[thr].get(clsname, {})
                    for attr_name, value_results in subgroups.items():
                        for value, sub_result in value_results.items():
                            # 构建键名：SD_Line_Map_{metric}/{clsname}_{attrname}_{value}_AP_thr_{thr}
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
        all_image_arranged_pred_results = []
        all_image_arranged_gt_annotations = []

        for pred_result, gt_annotation in zip(results.values(), annotations.values()):
            single_image_pred_results = []
            single_image_gt_annotations = []

            # ---------------- preds ----------------
            for instance in pred_result['pred_instances']:
                instance_data = [[int(x[0]), int(x[1])] for x in instance['data']]
                single_image_pred_results.append({
                    'pts': instance_data,
                    'type': instance['class'],
                    'attrs': instance['attrs'],
                    'confidence_level': instance['confidence_level']
                })

            # ---------------- gts ----------------
            for instance in gt_annotation['instances']:
                instance_cat = instance['category']
                instance_data = instance['position']
                instance_attrs = instance['attributes']

                # 统一构建5维属性索引（与TrunkVectorizedLocalMap保持一致）
                if instance_cat == 'lane_line':
                    # 车道线：线型(0)、功能(1)、颜色(2)、路沿(3)、停止线(4) 
                    gt_attrs = [
                        self.attrs_dict["laneline_linetype"].index(instance_attrs["laneline_linetype"]),
                        self.attrs_dict["laneline_function"].index(instance_attrs["laneline_function"]),
                        self.attrs_dict["laneline_color"].index(instance_attrs["laneline_color"]),
                        self.attrs_dict["curb_linetype"].index("unknown"),
                        self.attrs_dict["stopline_linetype"].index("unknown")
                    ]
                elif instance_cat == 'curb':
                    # 路沿：线型(0)、功能(1)、颜色(2)、路沿(3)、停止线(4)
                    gt_attrs = [
                        self.attrs_dict["laneline_linetype"].index("no"),
                        self.attrs_dict["laneline_function"].index("no"),
                        self.attrs_dict["laneline_color"].index("no"),
                        self.attrs_dict["curb_linetype"].index(instance_attrs["curb_linetype"]),
                        self.attrs_dict["stopline_linetype"].index("unknown")
                    ]
                elif instance_cat == 'stop_line':
                    # 停止线：线型(0)、功能(1)、颜色(2)、路沿(3)、停止线(4)
                    gt_attrs = [
                        self.attrs_dict["laneline_linetype"].index("no"),
                        self.attrs_dict["laneline_function"].index("no"),
                        self.attrs_dict["laneline_color"].index("no"),
                        self.attrs_dict["curb_linetype"].index("unknown"),
                        self.attrs_dict["stopline_linetype"].index(instance_attrs["stopline_linetype"])
                    ]
                else:
                    continue

                single_image_gt_annotations.append({
                    'pts': instance_data,
                    'type': self.MAPCLASSES.index(instance_cat),
                    'attrs': gt_attrs  # 使用统一5维属性
                })

            all_image_arranged_pred_results.append(single_image_pred_results)
            all_image_arranged_gt_annotations.append(single_image_gt_annotations)

        return all_image_arranged_pred_results, all_image_arranged_gt_annotations

    
    def show_result_with_original_image(self,
                                        pred_instances,
                                        out_dir,
                                        image_path):
        path_parts = image_path.split('/')
        bag_folder = None
        for part in path_parts:
            if '.bag' in part:
                bag_folder = part
                break
        if bag_folder:
            out_subdir = os.path.join(out_dir, bag_folder)
            os.makedirs(out_subdir, exist_ok=True)
        else:
            out_subdir = out_dir
        image_name = image_path.split('/')[-1]
        image = cv.imread(image_path)
        # 修正：类别颜色映射（保持不变，可按需调整）
        color = {0:(0, 0, 0), 1:(0, 0, 255), 2:((0, 97,255))}
        
        # 修正：attrs_map替换为新标注分类
        attrs_map = [
            # 0: lane_line（线型）- 7类细分类 + 占位值
            ["single_solid", "single_dash", "double_left_solid", "double_right_solid",
            "double_solid", "double_dash", "colored_three_line", "other", "unknown", "no"],
            # 1: curb（路沿类型）- 7类细分类 + 占位值
            ["road_edge", "plain_edge", "cone_edge", "waterhorse_edge",
            "fence_edge", "park_edge", "other", "unknown"],
            # 2: stop_line（停止线类型）
            ["normal", "other", "unknown"]
        ]
        
        for pred_instance in pred_instances:
            sub_pred_points = pred_instance['data']
            pred_class = pred_instance['class']
            pred_attr = pred_instance['attrs']
            
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
                # 修正：属性索引逻辑（匹配新attrs_map）
                if pred_class == 0:  # lane_line
                    text = attrs_map[0][pred_attr[0]]
                elif pred_class == 1:  # curb
                    text = attrs_map[1][pred_attr[3]]
                elif pred_class == 2:  # stop_line
                    text = attrs_map[2][pred_attr[4]]
                    
                ypos=sub_pred_points[0][1]-15
                if sub_pred_points[0][1]<10:
                    ypos=sub_pred_points[0][1]+15
                cv.putText(
                    image,
                    text,
                    (sub_pred_points[0][0]-5, ypos),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    3,
                    cv.LINE_AA
                )
                cv.putText(
                    image,
                    text,
                    (sub_pred_points[0][0]-5, ypos),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA
                )

        original_image = cv.imread(image_path)
        drew_image = cv.copyMakeBorder(image, 0, 0, 50, 0, cv.BORDER_CONSTANT, value=(128,128,128))
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
# 替换原有attrs_dict，完全匹配新标注规范
    attrs_dict = dict(
        # 车道线线型：7类新核心分类（无旧分类thick_dash）
        laneline_linetype=[
            "single_solid", 
            "single_dash", 
            "double_left_solid", 
            "double_right_solid", 
            "double_solid", 
            "double_dash", 
            "colored_three_line",
            "other", 
            "unknown", 
            "no"
        ],
        # 车道线功能：5类新核心分类（替代原有laneline_shape）
        laneline_function=[
            "normal",
            "fishbone_line",
            "stop_line",
            "diversion_line",
            "guide_line",
            "other",
            "no"
        ],
        # 车道线颜色：3类新分类
        laneline_color=[
            "white",
            "yellow",
            "other",
            "no"
        ],
        # 路沿类型：7类新核心分类（无旧分类road_boundary/cone_boundary）
        curb_linetype=[
            "road_edge",
            "plain_edge",
            "cone_edge",
            "waterhorse_edge",
            "fence_edge",
            "park_edge",
            "other",
            "unknown"
        ],
        # 停止线类型：保持不变（兼容映射）
        stopline_linetype=[
            "normal",
            "other",
            "unknown"
        ]
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
