import scipy
import numpy as np
from os import path as osp
from PIL import Image
import torch
from tqdm import tqdm
from  multiprocessing import Pool
from functools import partial
from sklearn import metrics
import warnings
import json
warnings.filterwarnings("ignore")
from shapely.geometry import LineString
from .mean_ap import average_precision
from .tpfp import tpfp_gen, custom_tpfp_gen_1
import mmcv
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from mmcv.utils import print_log
from terminaltables import AsciiTable
from collections import defaultdict
from functools import partial
import multiprocessing

def get_cls_results(gen_results, 
                    annotations,
                    num_sample=100, 
                    num_pred_pts_per_instance=30,
                    eval_use_same_gt_sample_num_flag=False,
                    class_id=0, 
                    fix_interval=False):
    """Get det results and gt information of a certain class.

    Args:
        gen_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes
    """
    # if len(gen_results) == 0 or 
    # import pdb;pdb.set_trace()
    # print(len(gen_results))
    # print(len(annotations))
    cls_gens, cls_scores, gen_attrs = [], [], []
    for res in gen_results:
        if res['type'] == class_id:
            if len(res['pts']) < 2:
                continue
            if not eval_use_same_gt_sample_num_flag:
                sampled_points = np.array(res['pts'])
            else:
                line = res['pts']
                line = LineString(line)

                if fix_interval:
                    distances = list(np.arange(1., line.length, 1.))
                    distances = [0,] + distances + [line.length,]
                    sampled_points = np.array([list(line.interpolate(distance).coords)
                                            for distance in distances]).reshape(-1, 2)
                else:
                    distances = np.linspace(0, line.length, num_sample)
                    sampled_points = np.array([list(line.interpolate(distance).coords)
                                                for distance in distances]).reshape(-1, 2)
            cls_gens.append(sampled_points)
            cls_scores.append(res['confidence_level'])
            gen_attrs.append(res['attrs'])

    num_res = len(cls_gens)
    if num_res > 0:
        cls_gens = np.stack(cls_gens).reshape(num_res,-1)
        cls_scores = np.array(cls_scores)[:,np.newaxis]
        cls_gens = np.concatenate([cls_gens,cls_scores],axis=-1)
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')
    else:
        if not eval_use_same_gt_sample_num_flag:
            cls_gens = np.zeros((0,num_pred_pts_per_instance*2+1))
        else:
            cls_gens = np.zeros((0,num_sample*2+1))
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')

    cls_gts, gt_attrs = [], []
    for ann in annotations:
        if ann['type'] == class_id:
            # line = ann['pts'] +  np.array((1,1)) # for hdmapnet
            line = ann['pts']
            # line = ann['pts'].cumsum(0)
            line = LineString(line)
            distances = np.linspace(0, line.length, num_sample)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                        for distance in distances]).reshape(-1, 2)
            
            cls_gts.append(sampled_points)
            gt_attrs.append(ann['attrs'])
    num_gts = len(cls_gts)
    if num_gts > 0:
        cls_gts = np.stack(cls_gts).reshape(num_gts,-1)
    else:
        cls_gts = np.zeros((0,num_sample*2))
    return cls_gens, cls_gts, gen_attrs, gt_attrs
    # ones = np.ones((num_gts,1))
    # tmp_cls_gens = np.concatenate([cls_gts,ones],axis=-1)
    # return tmp_cls_gens, cls_gts

def format_res_gt_by_classes(result_path,
                             gen_results,
                             annotations,
                             cls_names=None,
                             num_pred_pts_per_instance=30,
                             eval_use_same_gt_sample_num_flag=False,
                             nproc=24):
    assert cls_names is not None
    timer = mmcv.Timer()
    fix_interval = False
    print('results path: {}'.format(result_path))
    
    output_dir = osp.join(*osp.split(result_path)[:-1])
    # import pdb;pdb.set_trace()
    assert len(gen_results) == len(annotations)

    pool = Pool(nproc)
    cls_gens, cls_gts = {}, {}
    print('Formatting ...')
    formatting_file = 'cls_formatted.pkl'
    formatting_file = osp.join(output_dir,formatting_file)
    # arranged_pred_results, arranged_gt_annotations = arrange_results_and_annotations(gen_results, annotations)
    num_fixed_sample_pts = 100
    for i, clsname in enumerate(cls_names):
        # import pdb;pdb.set_trace()
        # for gen_result, annotation in zip(gen_results, annotations):
        #     gengts = get_cls_results(gen_result,
        #                     annotation,
        #                     num_sample=num_fixed_sample_pts,
        #                     num_pred_pts_per_instance=num_pred_pts_per_instance,
        #                     eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        #                     class_id=i,
        #                     fix_interval=fix_interval)
            # gens, gts, gens_attrs, gts_attrs = tuple(zip(*gengts))
        gengts = pool.starmap(
                partial(get_cls_results, num_sample=num_fixed_sample_pts,
                    num_pred_pts_per_instance=num_pred_pts_per_instance,
                    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
                    class_id=i,
                    fix_interval=fix_interval),
                zip(gen_results, annotations))   

        gens, gts, gens_attrs, gts_attrs = tuple(zip(*gengts))    
        cls_gens[clsname] = {'data':gens, 'attrs':gens_attrs}
        cls_gts[clsname] = {'data':gts, 'attrs':gts_attrs}
    # import pdb;pdb.set_trace()
    mmcv.dump([cls_gens, cls_gts],formatting_file)
    print('Cls data formatting done in {:2f}s!! with {}'.format(float(timer.since_start()),formatting_file))
    pool.close()
    return cls_gens, cls_gts

def eval_map(gen_results,
             annotations,
             cls_gens,
             cls_gts,
             threshold=0.5,
             cls_names=None,
             attrs_dict=None,
             logger=None,
             tpfp_fn=None,
             metric=None,
             num_pred_pts_per_instance=30,
             nproc=24):
    class_to_attrs = {
        'lane_line': ['laneline_linetype', 'laneline_shape'],  # lane_line的属性
        'curb': ['curb_linetype'],  # curb的属性
        'stop_line': ['stopline_linetype']  # stop_line的属性
    }
    timer = mmcv.Timer()
    pool = Pool(nproc)

    eval_results = []
    for clsname in cls_names:  # 为每个类别初始化一个结果字典
        # 预填充基础字段，避免后续访问时出现 KeyError 或 IndexError
        eval_results.append({
            'num_gts': 0,           # 该类别的真值数量
            'num_dets': 0,          # 该类别的预测数量
            'recall': np.array([]), # 召回率数组
            'precision': np.array([]), # 精确率数组
            'ap': 0.0,              # 平均精度
            'attrs_metric': {},     # 属性总体指标
            'attr_subgroups': {}    # 属性子组（按取值细分）指标
        })
    
    for i, clsname in enumerate(cls_names):
        # 从 cls_gens 中获取当前类别的预测数据和属性
        cls_gen_data = cls_gens[clsname]['data']  # 预测点数据（按图像分组）
        cls_gen_attrs = cls_gens[clsname]['attrs']  # 预测属性（按图像分组）
        # 从 cls_gts 中获取当前类别的真值数据和属性
        cls_gt_data = cls_gts[clsname]['data']  # 真值点数据（按图像分组）
        cls_gt_attrs = cls_gts[clsname]['attrs']  # 真值属性（按图像分组）
        # 1. 获取当前类别的属性列表（如lane_line对应['laneline_linetype', 'laneline_shape']）
        attrs = class_to_attrs.get(clsname, [])
        eval_results[i]['attr_subgroups'] = {}  # 存储属性子组结果

        for attr_name in attrs:
            eval_results[i]['attr_subgroups'][attr_name] = {}  # 存储该属性的所有取值结果
            attr_values = attrs_dict[attr_name]  # 该属性的所有可能值（如['single_solid', ...]）
            # 确定该属性在属性列表中的索引（如lane_line的laneline_linetype是第0个属性）
            attr_idx_gen = {
                'lane_line': {'laneline_linetype': 0, 'laneline_shape': 1},
                'curb': {'curb_linetype': 2},
                'stop_line': {'stopline_linetype': 3}
            }[clsname][attr_name]
            attr_idx_gt = {
                'lane_line': {'laneline_linetype': 0, 'laneline_shape': 1},
                'curb': {'curb_linetype': 0},
                'stop_line': {'stopline_linetype': 0}
            }[clsname][attr_name]

            # 2. 遍历该属性的每个取值（如single_solid、single_dash）
            for v in attr_values:
                v_code = attr_values.index(v)  # 取值对应的编码（如single_solid对应0）

                # 3. 筛选预测数据：保留该属性值为v_code的样本
                cls_gen_data_sub = []  # 子组预测数据（按图像分组）
                cls_gen_attrs_sub = []  # 子组预测属性
                for img_gen_data, img_gen_attrs in zip(cls_gen_data, cls_gen_attrs):
                    mask = [attr[attr_idx_gen] == v_code for attr in img_gen_attrs]
                    if len(mask) > 0 and np.any(mask):
                        # 有符合条件的预测数据，直接筛选
                        gen_data_sub = img_gen_data[mask]
                    else:
                        # 无符合条件的数据，生成空数组（指定列数与有效数据一致）
                        if img_gen_data.size == 0:
                            # 若原图数据为空，默认列数（根据实际场景修改，例如 201）
                            cols = 201
                        else:
                            # 从原图数据中获取列数（假设 img_gen_data 是二维数组）
                            cols = img_gen_data.shape[1]
                        gen_data_sub = np.empty((0, cols), dtype=img_gen_data.dtype)  # 空数组但列数正确
                    
                    gen_attrs_sub = [img_gen_attrs[k] for k in range(len(img_gen_attrs)) if mask[k]] if len(mask) > 0 else []
                    cls_gen_data_sub.append(gen_data_sub)
                    cls_gen_attrs_sub.append(gen_attrs_sub)

                # 4. 筛选真值数据：保留该属性值为v_code的样本
                cls_gt_data_sub = []  # 子组真值数据（按图像分组）
                cls_gt_attrs_sub = []  # 子组真值属性
                for img_gt_data, img_gt_attrs in zip(cls_gt_data, cls_gt_attrs):
                    # 筛选条件：真值属性中该属性的值为v_code
                    mask = [attr[attr_idx_gt] == v_code for attr in img_gt_attrs]
                    gt_data_sub = img_gt_data[mask] if len(mask) > 0 else np.array([])
                    gt_attrs_sub = [img_gt_attrs[k] for k in range(len(img_gt_attrs)) if mask[k]]
                    cls_gt_data_sub.append(gt_data_sub)
                    cls_gt_attrs_sub.append(gt_attrs_sub)

                tpfp_fn = custom_tpfp_gen_1
                tpfp_fn = partial(tpfp_fn, threshold=threshold, metric=metric)
                args = []
                # 5. 计算子组的TP/FP（复用原有逻辑）
                tpfp_sub = pool.starmap(
                    tpfp_fn,
                    zip(cls_gen_data_sub, cls_gt_data_sub)  # 传入筛选后的子组数据
                )
                tp_sub, fp_sub, _, _ = tuple(zip(*tpfp_sub))

                # 6. 计算子组的AP、Recall、Precision
                num_gts_sub = sum(bbox.shape[0] for bbox in cls_gt_data_sub)
                cls_gen_all_sub = np.vstack(cls_gen_data_sub)
                num_dets_sub = cls_gen_all_sub.shape[0]

                # 排序（按置信度降序）
                if num_dets_sub > 0:
                    sort_inds_sub = np.argsort(-cls_gen_all_sub[:, -1])
                    tp_sorted_sub = np.hstack(tp_sub)[sort_inds_sub]
                    fp_sorted_sub = np.hstack(fp_sub)[sort_inds_sub]
                    # 累积TP/FP并计算指标
                    tp_cum_sub = np.cumsum(tp_sorted_sub)
                    fp_cum_sub = np.cumsum(fp_sorted_sub)
                    eps = np.finfo(np.float32).eps
                    recalls_sub = tp_cum_sub / np.maximum(num_gts_sub, eps)
                    precisions_sub = tp_cum_sub / np.maximum(tp_cum_sub + fp_cum_sub, eps)
                    ap_sub = average_precision(recalls_sub, precisions_sub, mode='area')
                else:
                    recalls_sub = np.array([0.0]) if num_gts_sub > 0 else np.array([])
                    precisions_sub = np.array([0.0]) if num_gts_sub > 0 else np.array([])
                    ap_sub = 0.0 if num_gts_sub > 0 else np.nan  # 无预测时AP为0（若有真值）

                # 7. 存储子组结果
                eval_results[i]['attr_subgroups'][attr_name][v] = {
                    'num_gts': num_gts_sub,
                    'num_dets': num_dets_sub,
                    'ap': ap_sub,
                    'recall': recalls_sub,
                    'precision': precisions_sub
                }
        
        # 1. 汇总所有属性子组的 gts 和 dets，作为总类别的 num_gts 和 num_dets
        total_gts = 0
        total_dets = 0
        attr_subgroups = eval_results[i]['attr_subgroups']
        attr_name = next(iter(attr_subgroups.keys()))
        for v in eval_results[i]['attr_subgroups'][attr_name]:
            total_gts += eval_results[i]['attr_subgroups'][attr_name][v]['num_gts']
            total_dets += eval_results[i]['attr_subgroups'][attr_name][v]['num_dets']
        eval_results[i]['num_gts'] = total_gts
        eval_results[i]['num_dets'] = total_dets

        # 2. 计算总类别的 TP/FP（基于该类别的全部数据，不按属性筛选）
        if total_gts == 0 and total_dets == 0:
            # 无数据时直接赋值空指标
            eval_results[i]['recall'] = np.array([])
            eval_results[i]['precision'] = np.array([])
            eval_results[i]['ap'] = 0.0
        else:
            # 计算全部预测数据的 TP/FP
            tpfp_fn = partial(custom_tpfp_gen_1, threshold=threshold, metric=metric)
            tpfp_total = pool.starmap(tpfp_fn, zip(cls_gen_data, cls_gt_data))  # 用全部数据计算
            tp_total, fp_total, _, _ = tuple(zip(*tpfp_total))

            # 3. 计算总类别的 recall、precision、ap
            cls_gen_all_total = np.vstack(cls_gen_data) if cls_gen_data else np.array([])
            num_dets_total = cls_gen_all_total.shape[0]

            if num_dets_total > 0:
                # 按置信度排序
                sort_inds_total = np.argsort(-cls_gen_all_total[:, -1])
                tp_sorted_total = np.hstack(tp_total)[sort_inds_total]
                fp_sorted_total = np.hstack(fp_total)[sort_inds_total]

                # 累积 TP/FP
                tp_cum_total = np.cumsum(tp_sorted_total)
                fp_cum_total = np.cumsum(fp_sorted_total)
                eps = np.finfo(np.float32).eps
                recalls_total = tp_cum_total / np.maximum(total_gts, eps)
                precisions_total = tp_cum_total / np.maximum(tp_cum_total + fp_cum_total, eps)
                ap_total = average_precision(recalls_total, precisions_total, mode='area')
            else:
                # 无预测时的指标
                recalls_total = np.array([0.0]) if total_gts > 0 else np.array([])
                precisions_total = np.array([0.0]) if total_gts > 0 else np.array([])
                ap_total = 0.0 if total_gts > 0 else np.nan

            # 赋值总类别指标
            eval_results[i]['recall'] = recalls_total
            eval_results[i]['precision'] = precisions_total
            eval_results[i]['ap'] = ap_total

        print('cls:{} done in {:2f}s!!'.format(clsname,float(timer.since_last_check())))
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if len(aps) else 0.0

    print_map_summary(
        mean_ap, eval_results, class_name=cls_names, attrs_dict= attrs_dict, logger=logger)

    return mean_ap, eval_results 

def print_map_summary(mean_ap,
                      results,
                      class_name=None,
                      attrs_dict=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class, including attribute subgroups (不依赖AsciiTable)."""
    if logger == 'silent':
        return

    # 处理多尺度评估（保持原有逻辑）
    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1
    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            precisions[:, i] = np.array(cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    label_names = class_name
    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    # 定义表格生成函数（核心：用字符串拼接表格）
    def generate_table(table_data):
        """
        手动生成ASCII表格字符串
        Args:
            table_data: 二维列表，格式为[表头行, 数据行1, 数据行2, ...]
        Returns:
            str: 可直接打印的表格字符串
        """
        if not table_data:
            return ""
        
        # 计算每列的最大宽度（用于对齐）
        col_widths = []
        for col_idx in range(len(table_data[0])):
            max_len = max(len(str(row[col_idx])) for row in table_data)
            col_widths.append(max_len)
        
        # 生成分隔线（如"+-------+-------+"）
        separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
        
        # 生成每行内容（如"| class | gts  |"）
        rows = [separator]
        for row in table_data:
            formatted_cells = []
            for idx, cell in enumerate(row):
                # 单元格内容左右对齐（预留2个空格的边距）
                formatted_cell = f" {str(cell):<{col_widths[idx]}} "
                formatted_cells.append(formatted_cell)
            rows.append("|" + "|".join(formatted_cells) + "|")
            # 表头下方添加分隔线
            if rows.index("|" + "|".join(formatted_cells) + "|") == 1:
                rows.append(separator)
        
        rows.append(separator)
        return "\n".join(rows)

    # 遍历每个尺度（保持原有逻辑）
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'\nScale range {scale_ranges[i]}', logger=logger)
        
        # 1. 打印类别总体指标
        class_table_data = [
            ['class', 'gts', 'dets', 'precision', 'recall', 'ap']  # 表头
        ]
        for j in range(num_classes):
            class_table_data.append([
                label_names[j], 
                num_gts[i, j], 
                results[j]['num_dets'],
                f'{precisions[i, j]:.3f}', 
                f'{recalls[i, j]:.3f}', 
                f'{aps[i, j]:.3f}'
            ])
        class_table_data.append(['mAP', '', '', '', '', f'{mean_ap[i]:.3f}'])
        # 生成并打印表格
        class_table_str = generate_table(class_table_data)
        print_log('\n类别总体指标:', logger=logger)
        print_log(class_table_str, logger=logger)

        # 2. 打印属性总体指标
        for j in range(num_classes):
            clsname = label_names[j]
            attrs_metric = results[j].get('attrs_metric', {})
            if not attrs_metric:
                continue
            attrs_table_data = [
                ['attribute', 'precision', 'recall', 'f1_score']  # 表头
            ]
            for attr_name, metric in attrs_metric.items():
                attrs_table_data.append([
                    attr_name,
                    f"{metric['precision']:.3f}",
                    f"{metric['recall']:.3f}",
                    f"{metric['f1_score']:.3f}"
                ])
            attrs_table_str = generate_table(attrs_table_data)
            print_log(f'\n{clsname} 属性总体指标:', logger=logger)
            print_log(attrs_table_str, logger=logger)

        # 3. 打印属性子组（按取值细分）指标
        for j in range(num_classes):
            clsname = label_names[j]
            attr_subgroups = results[j].get('attr_subgroups', {})
            if not attr_subgroups:
                continue
            
            for attr_name, value_results in attr_subgroups.items():
                subgroup_table_data = [
                    ['value', 'gts', 'dets', 'precision', 'recall', 'ap']  # 表头
                ]
                for value, sub_result in value_results.items():
                    precision = sub_result['precision'][-1] if len(sub_result['precision'])>0 else 0.0
                    recall =  sub_result['recall'][-1] if len(sub_result['recall'])>0 else 0.0
                    subgroup_table_data.append([
                        value,
                        sub_result['num_gts'],
                        sub_result['num_dets'],
                        f"{precision:.3f}",
                        f"{recall:.3f}",
                        f"{sub_result['ap']:.3f}"
                    ])
                subgroup_table_str = generate_table(subgroup_table_data)
                print_log(f'\n{clsname} 属性 "{attr_name}" 细分指标:', logger=logger)
                print_log(subgroup_table_str, logger=logger)