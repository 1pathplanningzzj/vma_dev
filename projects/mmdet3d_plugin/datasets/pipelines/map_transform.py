import numpy as np
import torch
from shapely.geometry import LineString
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.builder import PIPELINES

from ..sd_driving_line_dataset import InstanceLines, VectorizedLocalMap

class TrunkInstanceLines(InstanceLines):
    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance, instance_class, instance_attrs in zip(self.instance_list, self.instance_labels, self.instance_attrs):
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    # import pdb;pdb.set_trace()
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                # if instance_class != self.map_classes.index('lane') and instance_attrs[list(self.attrs_dict.keys()).index('lane_direction')] != self.attrs_dict['lane_direction'].index('unidirectional'):
                #     flip_sampled_points = np.flip(sampled_points, axis=0)
                #     shift_pts_list.append(sampled_points)
                #     shift_pts_list.append(flip_sampled_points)
                # else:
                #     shift_pts_list.append(sampled_points)
                # !: Here we simplify it to only consider one direction for all insts
                shift_pts_list.append(sampled_points)
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)

            multi_shifts_pts_tensor[:,:,0] /= self.max_x # normalize
            multi_shifts_pts_tensor[:,:,1] /= self.max_y
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=0,max=0.999)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=0,max=0.999)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num - multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        
        return instances_tensor

class TrunkVectorizedLocalMap(VectorizedLocalMap):
    
    def gen_vectorized_samples(self, instances, attrs_dict):
        vectors = []
        for instance in instances:
            instance_cat = instance['category']
            instance_data = instance['position']
            instance_attrs = instance['attributes']

            if instance_cat == "lane_line":
                # lane_line：线型 / 功能 / 颜色
                attr_idx = [
                    attrs_dict["laneline_linetype"].index(instance_attrs["laneline_linetype"]),
                    attrs_dict["laneline_function"].index(instance_attrs["laneline_function"]),
                    attrs_dict["laneline_color"].index(instance_attrs["laneline_color"]),
                    attrs_dict["curb_linetype"].index("other"),
                    attrs_dict["stopline_linetype"].index("other"),
                ]

            elif instance_cat == "curb":
                attr_idx = [
                    attrs_dict["laneline_linetype"].index("other"),
                    attrs_dict["laneline_function"].index("no"),  
                    attrs_dict["laneline_color"].index("other"),
                    attrs_dict["curb_linetype"].index(instance_attrs["curb_linetype"]),
                    attrs_dict["stopline_linetype"].index("other"),
                ]

            elif instance_cat == "stop_line":
                # stop_line：修正laneline_function占位为no（匹配预处理逻辑）
                attr_idx = [
                    attrs_dict["laneline_linetype"].index("other"),
                    attrs_dict["laneline_function"].index("no"),  # 修正：从normal改为no
                    attrs_dict["laneline_color"].index("other"),
                    attrs_dict["curb_linetype"].index("other"),
                    attrs_dict["stopline_linetype"].index(instance_attrs["stopline_linetype"]),
                ]

            else:
                continue

            vectors.append((
                LineString(np.array(instance_data)),
                self.vec_classes.index(instance_cat),
                attr_idx
            ))
            
        gt_labels = []
        gt_instances = []
        gt_attrs = []
        for instance, instance_type, instance_attr in vectors:
            if instance_type != -1:
                gt_instances.append(instance)
                gt_labels.append(instance_type)
                gt_attrs.append(instance_attr)
        gt_instances = TrunkInstanceLines(self.vec_classes, attrs_dict, gt_instances, gt_labels, gt_attrs, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num,self.padding_value, patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instances,
            gt_vecs_label=gt_labels,
            gt_vecs_attr=gt_attrs,
        )
        return anns_results

@PIPELINES.register_module() 
class VectorizeMap(object):
    def __init__(
        self, 
        input_shape, 
        map_classes,
        attrs_dict,
        fixed_ptsnum_per_line, 
        padding_value=-1000
    ):
        self.vectormap = TrunkVectorizedLocalMap(
            patch_size=input_shape[::-1], # (h, w)
            map_classes=map_classes, 
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
        )
        self.attrs_dict = attrs_dict
    
    def __call__(self, results):
        instances = results["instances"]
        anns_results = self.vectormap.gen_vectorized_samples(instances, self.attrs_dict)
        results['gt_vecs_label'] = anns_results['gt_vecs_label']
        results['gt_vecs_attr'] = anns_results['gt_vecs_attr']
        results['gt_vecs_pts_loc'] = anns_results['gt_vecs_pts_loc']
        return results
