import os
import sys
import json
import argparse
import time
import torch
import mmcv
import logging
import numpy as np
from flask import send_file
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from pathlib import Path
# 获取上级目录路径（推荐使用 Path，兼容不同操作系统）
parent_dir = str(Path(__file__).resolve().parent.parent)

# 将上级目录添加到 sys.path
sys.path.append(parent_dir)
import tempfile
import cv2 as cv
import numpy as np
from mmcv import Config
from mmdet3d.apis import init_model
from mmdet3d.models import build_model
from mmcv.parallel import MMDataParallel
from mmdet.datasets import replace_ImageToTensor
from mmcv.runner import load_checkpoint
from mmdet3d.datasets.pipelines import Compose
from projects.mmdet3d_plugin.datasets.pipelines import LoadImageFromFiles,NormalizeImage,PadChannel
from shapely.geometry import LineString
#from data_generator.generator.vma_auto_generator import VMAAutoGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局变量：模型、配置、预处理管道
model = None
cfg = None
test_pipeline = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lock = torch.multiprocessing.Lock()  # 线程安全锁


def setup_multi_processes(cfg):
    """设置多进程参数（单样本推理简化版）"""
    if hasattr(cfg, 'mp_cfg'):
        omp_num_threads = cfg.mp_cfg.get('omp_num_threads', None)
        if omp_num_threads is not None:
            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    logger.info("单样本推理模式：多进程设置初始化完成") # mp_cfg 是 并行编程的缓解变量 


def init_model_and_pipeline(config_path, checkpoint_path):
    """初始化模型和预处理管道（匹配TrunkLineDataset的处理逻辑）"""
    global cfg, test_pipeline 
    # 加载配置
    cfg = Config.fromfile(config_path) # 
    setup_multi_processes(cfg)         # 

    # 构建预处理管道（复用配置中的test pipeline，替换ImageToTensor）
    test_pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    logger.info("预处理管道初始化完成")

    # 构建模型
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 加载权重
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    # 融合conv和bn（加速推理）
    if hasattr(cfg, 'fuse_conv_bn') and cfg.fuse_conv_bn:
        from mmcv.cnn import fuse_conv_bn
        model = fuse_conv_bn(model)
    model.load_state_dict(checkpoint['state_dict'])
    # 设置类别信息（与TrunkLineDataset一致）
    model.CLASSES = ['lane_line', 'curb', 'stop_line']  # 必须与数据集类别匹配
    # 模型移至设备并设置为评估模式
    model = model.to(device).eval()
    logger.info("模型加载完成（适配TrunkLineDataset）")
    return model

def extract_datacontainer (data):
    """递归提取 DataContainer 中的原始数据"""
    if isinstance (data, mmcv.parallel.DataContainer):
        return data.data # 提取内部原始数据
    elif isinstance (data, dict):
        return {k: extract_datacontainer (v) for k, v in data.items ()}
    elif isinstance (data, list):
        return [extract_datacontainer (v) for v in data]
    else:
        return data

def preprocess_images(image_path, view_image_path=None):
    # 处理主图
    main_data = {'img_filename': image_path, 'test_mode': True}
    main_processed = test_pipeline(main_data)
    # 确保img是Tensor，img_metas是原生字典
    main_processed['img'] = torch.from_numpy(main_processed['img'])

    # 处理视角图
    view_processed = None
    if view_image_path and os.path.exists(view_image_path):
        view_data = {'img_filename': view_image_path, 'test_mode': True}
        view_processed = test_pipeline(view_data)
        view_processed['img'] = torch.from_numpy(view_processed['img'])

    # 构造输入数据（完全原生结构）
    input_data = {
        'img': main_processed['img'],
        'img_metas': extract_datacontainer(main_processed['img_metas']),  # 原生字典
        'view_img': view_processed['img'] if view_processed else None,
        'view_img_metas': extract_datacontainer(view_processed['img_metas']) if view_processed else None
    }
    print(input_data["img_metas"])
    return input_data


def infer_single_sample(model, input_data):
    """彻底移除所有DataContainer包装，使用纯原生数据结构"""
    with torch.no_grad():
        # 模型输入：img是带batch维度的Tensor，img_metas是列表套字典
        model_input = {
            'img': input_data['img'].unsqueeze(0).to(device),  # shape: [1, C, H, W]
            'img_metas': [input_data['img_metas']]  # 列表嵌套原生字典
        }
        # 处理视角图
        if input_data['view_img'] is not None:
            model_input['view_img'] = input_data['view_img'].unsqueeze(0).to(device)
            model_input['view_img_metas'] = [input_data['view_img_metas']]  # 同样列表套字典

        # 模型推理
        outputs = model(return_loss=False, **model_input)
    return outputs[0]

def convert_pred_to_native(pred_result):
    """
    将推理结果中的所有PyTorch张量转换为Python原生类型（列表/数值）
    
    Args:
        pred_result (dict): 模型输出的原始推理结果（包含张量）
        
    Returns:
        dict: 可JSON序列化的结果（仅包含列表、字典、数值等原生类型）
    """
    def _recursive_convert(data):
        # 处理PyTorch张量
        if isinstance(data, torch.Tensor):
            # 若为标量张量，转换为Python数值
            if data.numel() == 1:
                return data.item()
            # 若为数组张量，转换为列表
            else:
                return data.cpu().numpy().tolist()
        # 处理NumPy数组
        elif isinstance(data, np.ndarray):
            return data.tolist()
        # 处理字典
        elif isinstance(data, dict):
            return {k: _recursive_convert(v) for k, v in data.items()}
        # 处理列表/tuple
        elif isinstance(data, (list, tuple)):
            return [_recursive_convert(v) for v in data]
        # 其他原生类型（int/float/str等）直接返回
        else:
            return data
    
    return _recursive_convert(pred_result)

def show_result_on_bev_map(pred_result,img_path):
    imgname=img_path.split('/')[-1]
    image = cv.imread(img_path)
    color = {0:(0, 0, 0), 1:(0, 0, 255), 2:((0, 97,255))}
    attrs_map=[
        [ "single_solid", "single_dash", "double_solid", "double_dash", "thick_dash",
                    "other", "unknown", "no"],
        ["road_boundary", "cone_boundary", "other", "unknown"],
        ["normal", "other", "unknown"]
    ]
    
    # drew_image_path = out_dir + '/' + image_name
    # cv.imwrite(drew_image_path, image)
    res_dict=pred_result["pts_bbox"]
    pts=res_dict['pts_3d'].tolist()
    labels=res_dict['labels_3d'].tolist()
    attrs_dict=res_dict['attrs_3d']['attrs_preds']
    for j in range(len(pts)):
        # import pdb;pdb.set_trace()
        sub_pred_points = pts[j]
        pred_class = int(labels[j])
        
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
                attr=attrs_dict[0][j]
                text=attrs_map[pred_class][attr]
            else:
                attr=attrs_dict[2][j]
                text=attrs_map[pred_class][attr]
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

    original_image = cv.imread(img_path)
    drew_image = cv.copyMakeBorder(image, 0, 0, 50, 0, cv.BORDER_CONSTANT, value=(128,128,128))
    concat_image = cv.hconcat([original_image, drew_image])
    return concat_image
    


def format_result(pred_result,M_patch2bev,M_bev2world,pose):
    """格式化推理结果（匹配TrunkLineDataset的输出格式）"""
    formatted ={}
    res_dict=pred_result["pts_bbox"]
    pts=res_dict['pts_3d'].tolist()
    x,y,qx,qy,qz,qw=pose["x"],pose["y"],pose["qx"],pose["qy"],pose["qz"],pose["qw"]
    rotation_matrix = np.array([
    [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qw*qz, 0],
    [2*qx*qy+2*qw*qz, 1-2*qx*qx-2*qz*qz, 0],
    [0, 0, 1]
    ])
    translation_matrix = np.array([
    [1, 0, -x],
    [0, 1, -y],
    [0, 0, 1]
    ])
    M_world2ego=rotation_matrix@translation_matrix
    pts1=[]
    for line in pts:
        pts1.append(transform_line_coord_2d(line,M_patch2bev).tolist())
    pts2=[]
    for line in pts1:
        pts2.append(transform_line_coord_2d(line,M_bev2world).tolist())
    pts3=[]
    for line in pts2:
        pts3.append(transform_line_coord_2d(line,M_world2ego).tolist())
    labels=res_dict['labels_3d'].tolist()
    attrs_dict=res_dict['attrs_3d']['attrs_preds']
    Type=[]
    for i in range(len(labels)):
        if int(labels[i])==0:
            attr=attrs_dict[0][i]
            if attr==0 or attr==2:
                Type.append(1)
            elif attr==1 or attr==3:
                Type.append(2)
            else:
                Type.append(3)
        elif int(labels[i])==1:
            attr=attrs_dict[2][i]
            if attr<2:
                Type.append(1)
            else:
                Type.append(3)
        else:
            if attr<1:
                Type.append(1)
            else:
                Type.append(3)
    formatted["data"]=pts
    formatted["attr"]={"type":Type,
                       "class":labels}
    return formatted

def transform_line_coord_2d(line, matrix):
    """
    Args:
        line ( np.array or list): shape [N, 2]
        matrix (np.array): shape [N, 2]

    Returns:
        _type_: _description_
    """
    line = np.array(line)
    x, y = line[:, 0], line[:, 1]
    pts = np.vstack([x, y, np.ones_like(x)])
    pts_trans = matrix @ pts
    pts_trans = pts_trans[:2].T
    return pts_trans

@app.route('/infer', methods=['POST'])
def infer():
    """推理接口：接收主图和视角图，返回符合TrunkLineDataset格式的结果"""
    global model, test_pipeline, lock
    with lock:
        try:
            # 1. 接收上传的图片文件（主图必传，视角图可选）和转换参数矩阵参数矩阵
            if 'image' not in request.files:
                return jsonify({"error": "缺少主图参数：image"}), 400
            image_file = request.files['image']
            view_image_file = request.files.get('view_image')  # 视角图可选
            

            if image_file.filename == '':
                return jsonify({"error": "主图文件名为空"}), 400
            
            # 2. 正确接收表单中的JSON参数（关键修复）
            try:
                # 从form中获取JSON字符串并解析
                M_patch2bev_str = request.form.get('M_patch2bev')
                M_patch2bev = json.loads(M_patch2bev_str) if M_patch2bev_str else None

                M_bev2world_str = request.form.get('M_bev2world')
                M_bev2world = json.loads(M_bev2world_str) if M_bev2world_str else None

                pose_str = request.form.get('pose')
                pose = json.loads(pose_str) if pose_str else None
            except json.JSONDecodeError as e:
                return jsonify({"error": f"参数解析错误：{str(e)}"}), 400

            # 2. 保存图片到临时目录
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 保存主图
                main_img_path = os.path.join(tmp_dir, 'main_image.jpg')
                image_file.save(main_img_path)
                # 保存视角图（如果提供）
                view_img_path = None
                if view_image_file is not None and view_image_file.filename != '':
                    view_img_path = os.path.join(tmp_dir, 'view_image.jpg')
                    view_image_file.save(view_img_path)
                logger.info(f"图片保存至临时目录：{tmp_dir}（主图：{os.path.exists(main_img_path)}，视角图：{os.path.exists(view_img_path) if view_img_path else False}）")

                # 3. 预处理图片（匹配TrunkLineDataset的预处理逻辑）
                start_pre = time.time()
                input_data = preprocess_images(main_img_path, view_img_path)
                logger.info(f"图片预处理完成，耗时：{time.time() - start_pre:.2f}秒")

                # 4. 模型推理
                start_infer = time.time()
                pred_result = infer_single_sample(model, input_data)
                logger.info(f"推理完成，耗时：{time.time() - start_infer:.2f}秒")
                concat_img = show_result_on_bev_map(pred_result=pred_result,img_path=main_img_path)

                # 5. 格式化结果（匹配TrunkLineDataset的输出格式）
                formatted_result = format_result(pred_result,
                                                 M_patch2bev=M_patch2bev,
                                                 M_bev2world=M_bev2world,
                                                 pose=pose)

            # 6. 保存结果为JSON（路径可根据需求调整）
            '''result_dir = os.path.join(os.getcwd(), 'trunk_inference_results')
            os.makedirs(result_dir, exist_ok=True)
            result_filename = f"result_{int(time.time())}.json"
            result_path = os.path.join(result_dir, result_filename)
            with open(result_path, 'w') as f:
                json.dump({
                    'image_name': image_file.filename,
                    'pred_instances': formatted_result
                }, f, indent=2)
            logger.info(f"结果保存至：{result_path}")'''
            #concat_img_rgb = cv.cvtColor(concat_img, cv.COLOR_BGR2RGB)
            concat_img_rgb = concat_img
            cv.imwrite("vis.jpg", concat_img_rgb)
            # 编码为JPG格式
            _, img_encoded = cv.imencode('.jpg', concat_img_rgb)
            # 转为base64字符串
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            # 7. 返回结果
            return jsonify({
                "status": "success",
                "classes": model.CLASSES,
                "pred_instances": formatted_result,
                "vis_result":img_base64
            })

        except Exception as e:
            logger.error(f"推理出错：{str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='TrunkLineDataset对应的配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='模型权重文件路径')
    args = parser.parse_args()

    # 初始化模型和预处理管道（适配TrunkLineDataset）
    model = init_model_and_pipeline(args.config, args.checkpoint)

    # 启动Flask服务（关闭多线程，确保稳定性）
    app.run(host='0.0.0.0', port=8889, debug=True, threaded=False)
