import scipy
import numpy as np
from os import path as osp
from PIL import Image
import torch
from tqdm import tqdm
from multiprocessing import Pool
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
    else:
        if not eval_use_same_gt_sample_num_flag:
            cls_gens = np.zeros((0,num_pred_pts_per_instance*2+1))
        else:
            cls_gens = np.zeros((0,num_sample*2+1))

    cls_gts, gt_attrs = [], []
    for ann in annotations:
        if ann['type'] == class_id:
            line = ann['pts']
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
    assert len(gen_results) == len(annotations)

    pool = Pool(nproc)
    cls_gens, cls_gts = {}, {}
    print('Formatting ...')
    formatting_file = 'cls_formatted.pkl'
    formatting_file = osp.join(output_dir,formatting_file)
    num_fixed_sample_pts = 100
    for i, clsname in enumerate(cls_names):
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
             eval_use_same_gt_sample_num_flag=False,  # 补充该参数
             nproc=24):
    class_to_attrs = {
        'lane_line': ['laneline_linetype', 'laneline_function', 'laneline_color'],
        'curb': ['curb_linetype'],
        'stop_line': ['stopline_linetype']
    }
    
    timer = mmcv.Timer()
    pool = Pool(nproc)

    eval_results = []
    for clsname in cls_names:
        eval_results.append({
            'num_gts': 0,
            'num_dets': 0,
            'recall': np.array([]),
            'precision': np.array([]),
            'ap': 0.0,
            'attrs_metric': {},
            'attr_subgroups': {}
        })
    
    for i, clsname in enumerate(cls_names):
        cls_gen_data = cls_gens[clsname]['data']
        cls_gen_attrs = cls_gens[clsname]['attrs']
        cls_gt_data = cls_gts[clsname]['data']
        cls_gt_attrs = cls_gts[clsname]['attrs']
        # 获取当前类别的属性列表
        attrs = class_to_attrs.get(clsname, [])
        eval_results[i]['attr_subgroups'] = {}

        for attr_name in attrs:
            eval_results[i]['attr_subgroups'][attr_name] = {}
            attr_values = attrs_dict[attr_name]
            
            ATTR_ORDER = [
                'laneline_linetype',
                'laneline_function',
                'laneline_color',
                'curb_linetype',
                'stopline_linetype'
            ]

            attr_idx_gen = ATTR_ORDER.index(attr_name)
            attr_idx_gt  = ATTR_ORDER.index(attr_name)



            # 遍历属性每个取值
            for v in attr_values:
                v_code = attr_values.index(v)

                # 筛选预测数据
                cls_gen_data_sub = []
                cls_gen_attrs_sub = []
                for img_gen_data, img_gen_attrs in zip(cls_gen_data, cls_gen_attrs):
                    mask = [attr[attr_idx_gen] == v_code for attr in img_gen_attrs] if len(img_gen_attrs) > 0 else []
                    if len(mask) > 0 and np.any(mask):
                        gen_data_sub = img_gen_data[mask]
                    else:
                        # 优化3：容错空数据列数判断
                        if img_gen_data.size == 0:
                            if eval_use_same_gt_sample_num_flag:
                                cols = 100 * 2 + 1
                            else:
                                cols = num_pred_pts_per_instance * 2 + 1
                        else:
                            cols = img_gen_data.shape[1]
                        gen_data_sub = np.empty((0, cols), dtype=img_gen_data.dtype)
                    
                    gen_attrs_sub = [img_gen_attrs[k] for k in range(len(img_gen_attrs)) if mask[k]] if len(mask) > 0 else []
                    cls_gen_data_sub.append(gen_data_sub)
                    cls_gen_attrs_sub.append(gen_attrs_sub)

                # 筛选真值数据
                cls_gt_data_sub = []
                cls_gt_attrs_sub = []
                for img_gt_data, img_gt_attrs in zip(cls_gt_data, cls_gt_attrs):
                    mask = [attr[attr_idx_gt] == v_code for attr in img_gt_attrs] if len(img_gt_attrs) > 0 else []
                    gt_data_sub = img_gt_data[mask] if len(mask) > 0 and img_gt_data.size > 0 else np.array([])
                    gt_attrs_sub = [img_gt_attrs[k] for k in range(len(img_gt_attrs)) if mask[k]] if len(mask) > 0 else []
                    cls_gt_data_sub.append(gt_data_sub)
                    cls_gt_attrs_sub.append(gt_attrs_sub)

                # 计算TP/FP
                tpfp_fn = custom_tpfp_gen_1
                tpfp_fn = partial(tpfp_fn, threshold=threshold, metric=metric)
                tpfp_sub = pool.starmap(
                    tpfp_fn,
                    zip(cls_gen_data_sub, cls_gt_data_sub)
                )
                tp_sub, fp_sub, _, _ = tuple(zip(*tpfp_sub))

                # 计算子组指标
                num_gts_sub = sum(bbox.shape[0] for bbox in cls_gt_data_sub)
                # 容错非空数据拼接
                cls_gen_all_sub_list = [d for d in cls_gen_data_sub if d.size > 0]
                cls_gen_all_sub = np.vstack(cls_gen_all_sub_list) if cls_gen_all_sub_list else np.array([])
                num_dets_sub = cls_gen_all_sub.shape[0]

                if num_dets_sub > 0:
                    sort_inds_sub = np.argsort(-cls_gen_all_sub[:, -1])
                    tp_sorted_sub = np.hstack(tp_sub)[sort_inds_sub]
                    fp_sorted_sub = np.hstack(fp_sub)[sort_inds_sub]
                    tp_cum_sub = np.cumsum(tp_sorted_sub)
                    fp_cum_sub = np.cumsum(fp_sorted_sub)
                    eps = np.finfo(np.float32).eps
                    recalls_sub = tp_cum_sub / np.maximum(num_gts_sub, eps)
                    precisions_sub = tp_cum_sub / np.maximum(tp_cum_sub + fp_cum_sub, eps)
                    ap_sub = average_precision(recalls_sub, precisions_sub, mode='area')
                else:
                    recalls_sub = np.array([0.0]) if num_gts_sub > 0 else np.array([])
                    precisions_sub = np.array([0.0]) if num_gts_sub > 0 else np.array([])
                    ap_sub = 0.0 if num_gts_sub > 0 else np.nan

                # 存储子组结果
                eval_results[i]['attr_subgroups'][attr_name][v] = {
                    'num_gts': num_gts_sub,
                    'num_dets': num_dets_sub,
                    'ap': ap_sub,
                    'recall': recalls_sub,
                    'precision': precisions_sub
                }
        
        # 汇总总类别指标
        total_gts = 0
        total_dets = 0
        attr_subgroups = eval_results[i]['attr_subgroups']
        if attr_subgroups:
            total_gts = sum(
                gt.shape[0] for gt in cls_gt_data if gt.size > 0
            )

            total_dets = sum(
                det.shape[0] for det in cls_gen_data if det.size > 0
            )
        eval_results[i]['num_gts'] = total_gts
        eval_results[i]['num_dets'] = total_dets

        # 计算总类别TP/FP及指标
        if total_gts == 0 and total_dets == 0:
            eval_results[i]['recall'] = np.array([])
            eval_results[i]['precision'] = np.array([])
            eval_results[i]['ap'] = 0.0
        else:
            tpfp_fn = partial(custom_tpfp_gen_1, threshold=threshold, metric=metric)
            tpfp_total = pool.starmap(tpfp_fn, zip(cls_gen_data, cls_gt_data))
            tp_total, fp_total, _, _ = tuple(zip(*tpfp_total))

            # 优化4：容错总类别数据拼接
            cls_gen_all_total_list = [d for d in cls_gen_data if d.size > 0]
            cls_gen_all_total = np.vstack(cls_gen_all_total_list) if cls_gen_all_total_list else np.array([])
            num_dets_total = cls_gen_all_total.shape[0]

            if num_dets_total > 0:
                sort_inds_total = np.argsort(-cls_gen_all_total[:, -1])
                tp_sorted_total = np.hstack(tp_total)[sort_inds_total]
                fp_sorted_total = np.hstack(fp_total)[sort_inds_total]
                tp_cum_total = np.cumsum(tp_sorted_total)
                fp_cum_total = np.cumsum(fp_sorted_total)
                eps = np.finfo(np.float32).eps
                recalls_total = tp_cum_total / np.maximum(total_gts, eps)
                precisions_total = tp_cum_total / np.maximum(tp_cum_total + fp_cum_total, eps)
                ap_total = average_precision(recalls_total, precisions_total, mode='area')
            else:
                recalls_total = np.array([0.0]) if total_gts > 0 else np.array([])
                precisions_total = np.array([0.0]) if total_gts > 0 else np.array([])
                ap_total = 0.0 if total_gts > 0 else np.nan

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

    # 手动生成表格函数
    def generate_table(table_data):
        if not table_data:
            return ""
        
        col_widths = []
        for col_idx in range(len(table_data[0])):
            max_len = max(len(str(row[col_idx])) for row in table_data)
            col_widths.append(max_len)
        
        separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
        rows = [separator]
        for row in table_data:
            formatted_cells = []
            for idx, cell in enumerate(row):
                formatted_cell = f" {str(cell):<{col_widths[idx]}} "
                formatted_cells.append(formatted_cell)
            rows.append("|" + "|".join(formatted_cells) + "|")
            if rows.index("|" + "|".join(formatted_cells) + "|") == 1:
                rows.append(separator)
        
        rows.append(separator)
        return "\n".join(rows)

    # 打印类别总体指标
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'\nScale range {scale_ranges[i]}', logger=logger)
        
        class_table_data = [
            ['class', 'gts', 'dets', 'precision', 'recall', 'ap']
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
        class_table_str = generate_table(class_table_data)
        print_log('\n类别总体指标:', logger=logger)
        print_log(class_table_str, logger=logger)

        # 打印属性总体指标
        for j in range(num_classes):
            clsname = label_names[j]
            attrs_metric = results[j].get('attrs_metric', {})
            if not attrs_metric:
                continue
            attrs_table_data = [
                ['attribute', 'precision', 'recall', 'f1_score']
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

        # 打印属性子组指标
        for j in range(num_classes):
            clsname = label_names[j]
            attr_subgroups = results[j].get('attr_subgroups', {})
            if not attr_subgroups:
                continue
            
            for attr_name, value_results in attr_subgroups.items():
                subgroup_table_data = [
                    ['value', 'gts', 'dets', 'precision', 'recall', 'ap']
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