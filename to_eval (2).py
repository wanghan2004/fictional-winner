import sys
import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
from mmcv import Config
from types import SimpleNamespace
import yaml
import itertools
import math

# 注意：我们不再需要 YOLO，所以这里不再导入或使用它
# from ultralytics import YOLO 

# 引入你的自定义跟踪器
from custom_byte_tracker import ByteTracker, STrack

# ==============================================================================
# 1. 导入 Metric3D 模块
# ==============================================================================
print(">>> [DEBUG] 步骤 1: 导入 Metric3D 模块...")
METRIC3D_PATH = '/root/autodl-tmp/Metric3D'

if not os.path.exists(METRIC3D_PATH):
    print(f"!!! [WARNING] Metric3D 路径不存在: {METRIC3D_PATH}")
    # 尝试自动查找
    possible_paths = glob.glob('/root/autodl-tmp/**/Metric3D', recursive=True)
    if possible_paths:
        METRIC3D_PATH = possible_paths[0]
        print(f">>> [DEBUG] 找到 Metric3D 路径: {METRIC3D_PATH}")
    else:
        raise FileNotFoundError("Metric3D 目录未找到，请检查路径配置")

if METRIC3D_PATH not in sys.path:
    sys.path.insert(0, METRIC3D_PATH)

try:
    from mono.model.monodepth_model import DepthModel as MonoDepthModel
    print(">>> [INFO] Metric3D 模块导入成功。")
except ImportError as e:
    print(f"!!! [ERROR] 从 Metric3D 导入模块失败: {e}")
    raise

# ==============================================================================
# 2. 配置与路径定义
# ==============================================================================
print("\n>>> [DEBUG] 步骤 2: 配置模型和文件路径...")

# Metric3D 相关路径
METRIC3D_MODEL_PATH = '/root/autodl-tmp/weights/metric_depth_vit_large_800k.pth'
METRIC3D_CONFIG_PATH = os.path.join(METRIC3D_PATH, 'mono/configs/HourglassDecoder/vit.raft5.large.py')

# 视频输入与结果输出路径
INPUT_VIDEOS_DIR = '/root/autodl-tmp/kitti_videos'
BASE_OUTPUT_EVAL_DIR = '/root/autodl-tmp/eval_outputs3'
YAML_CONFIG_PATH = 'bytetrack.yaml'

# ★★★ 新增: RRC 检测结果的路径 ★★★
# 根据你之前的 ls 输出，这里指向 training 文件夹
RRC_DETECTIONS_DIR = '/root/autodl-tmp/DeepFusionMOT/data/detections/2D/rrc/training/Car'

# 验证关键路径
for path_name, path in [
    ("Metric3D 模型", METRIC3D_MODEL_PATH),
    ("Metric3D 配置文件", METRIC3D_CONFIG_PATH),
    ("输入视频目录", INPUT_VIDEOS_DIR),
    ("RRC 检测结果目录", RRC_DETECTIONS_DIR)
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path_name} 路径不存在: {path}")

os.makedirs(BASE_OUTPUT_EVAL_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> [INFO] 使用设备: {DEVICE}")

# ==============================================================================
# 3. 模型加载 (Metric3D)
# ==============================================================================
print("\n>>> [DEBUG] 步骤 3: 加载 Metric3D 模型...")

# 这里的全局变量用于存储加载好的模型
metric3d_model = None
cfg = None

try:
    cfg = Config.fromfile(METRIC3D_CONFIG_PATH)
    cfg.model.backbone.use_mask_token = False
    metric3d_model = MonoDepthModel(cfg).to(DEVICE)
    checkpoint = torch.load(METRIC3D_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    metric3d_model.load_state_dict(state_dict, strict=False)
    metric3d_model.eval()
    print(">>> [SUCCESS] Metric3Dv2 模型加载成功！")
except Exception as e:
    print(f"!!! [FATAL ERROR] 加载 Metric3Dv2 模型时出错: {e}")
    raise

# ==============================================================================
# 4. 辅助函数: 加载 KITTI 格式检测结果
# ==============================================================================
def load_kitti_detections(det_path, target_type='Car'):
    """
    加载 RRC 提供的简化版检测结果文件 (CSV 格式)
    文件格式: frame, x1, y1, x2, y2, score
    分隔符: 逗号
    """
    detections = {}
    if not os.path.exists(det_path):
        print(f"!!! [WARNING] 检测文件不存在: {det_path}")
        return detections

    with open(det_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 修复 1: 使用逗号分割
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()

            try:
                # 修复 2: 适配 6 列格式 (Frame, x1, y1, x2, y2, Score)
                # RRC 的这个文件里没有类别列，因为文件本身就在 Car 文件夹下，默认全都是 Car
                
                # 解析基础数据
                frame_idx = int(float(parts[0])) # 有时帧号可能是 0.0
                
                # 如果只有 6 列，直接读取坐标和分数
                if len(parts) >= 6:
                    x1 = float(parts[1])
                    y1 = float(parts[2])
                    x2 = float(parts[3])
                    y2 = float(parts[4])
                    score = float(parts[5])
                    
                    # 存储结果
                    if frame_idx not in detections:
                        detections[frame_idx] = []
                    detections[frame_idx].append([x1, y1, x2, y2, score])
                    
            except (ValueError, IndexError) as e:
                # print(f"[DEBUG] 解析错误行: {line} -> {e}") # 调试用
                continue
                
    return detections

# ==============================================================================
# 5. 视频处理主函数
# ==============================================================================
def process_video_for_eval(input_path, output_txt_path, tracker_args, detection_dir):
    video_basename = os.path.basename(input_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]
    
    print(f"\n--- 开始处理视频: {video_basename} ---")
    
    # 1. 准备检测数据
    # 假设检测文件名与视频文件名一致 (例如 0000.mp4 -> 0000.txt)
    det_file_path = os.path.join(detection_dir, f"{video_name_no_ext}.txt")
    print(f">>> [INFO] 加载检测文件: {det_file_path}")
    preloaded_dets = load_kitti_detections(det_file_path, target_type='Car')
    print(f">>> [INFO] 共加载了 {len(preloaded_dets)} 帧的检测数据")

    # 2. 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> [INFO] 视频详情: {width}x{height}, FPS={fps:.2f}, 总帧={total_frames}")

    # 3. 准备 Metric3D 输入尺寸
    if hasattr(cfg, 'data_basic') and 'vit_size' in cfg.data_basic:
        metric3d_input_size = (cfg.data_basic['vit_size'][1], cfg.data_basic['vit_size'][0])
    else:
        metric3d_input_size = (1024, 1024) # Fallback

    # 4. 初始化跟踪器
    depth_roi_scale = tracker_args.depth_roi_scale
    tracker = ByteTracker(args=tracker_args, frame_rate=fps)
    STrack.release_id()
    
    # 这里的 ID 设置为 0，因为我们只跟踪 Car
    TARGET_CLASS_ID = 0 
    TARGET_CLASS_NAME = 'Car'

    frame_count = 0
    
    # 5. 逐帧处理
    with open(output_txt_path, 'w') as f_out:
        with tqdm(total=total_frames, desc=f"Processing {video_basename}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # --- A. 获取当前帧的 2D 检测 (替换 YOLO) ---
                current_frame_dets = preloaded_dets.get(frame_count, [])

                # --- B. 深度估计 (Metric3D) ---
                with torch.no_grad():
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame_resized = cv2.resize(rgb_frame, metric3d_input_size)
                    rgb_torch = torch.from_numpy(rgb_frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
                    
                    pred_output = metric3d_model(data={'input': rgb_torch})
                    pred_depth_np = pred_output[0].squeeze().cpu().numpy()
                    pred_depth_filtered = cv2.resize(pred_depth_np, (width, height))

                # --- C. 融合检测框与深度信息 ---
                detections_with_depth = []
                if len(current_frame_dets) > 0:
                    for det in current_frame_dets:
                        x1, y1, x2, y2, score = det
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        
                        # 深度 ROI 采样
                        roi_w = max(1, int((x2 - x1) * depth_roi_scale))
                        roi_h = max(1, int((y2 - y1) * depth_roi_scale))
                        roi_x1 = max(0, int(x1 + ((x2 - x1) - roi_w) // 2))
                        roi_y1 = max(0, int(y1 + ((y2 - y1) - roi_h) // 2))
                        roi_x2 = min(width, roi_x1 + roi_w)
                        roi_y2 = min(height, roi_y1 + roi_h)
                        
                        depth_roi = pred_depth_filtered[roi_y1:roi_y2, roi_x1:roi_x2]
                        initial_depth = np.median(depth_roi) if depth_roi.size > 0 else 0.0
                        
                        detections_with_depth.append([x1, y1, x2, y2, score, TARGET_CLASS_ID, initial_depth])

                # --- D. 更新跟踪器 ---
                tracks = tracker.update(np.array(detections_with_depth)) if len(detections_with_depth) > 0 else np.empty((0, 8))

                # --- E. 写入结果 (KITTI 格式) ---
                if tracks.shape[0] > 0:
                    for track in tracks:
                        bb_left, bb_top, bb_right, bb_bottom = track[0], track[1], track[2], track[3]
                        track_id = int(track[4])
                        score = track[5]

                        # 格式: frame id type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry score
                        f_out.write(
                            f"{frame_count} {track_id} {TARGET_CLASS_NAME} -1 -1 -10 "
                            f"{bb_left:.2f} {bb_top:.2f} {bb_right:.2f} {bb_bottom:.2f} "
                            f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.4f}\n"
                        )

                frame_count += 1
                pbar.update(1)

    cap.release()
    print(f"--- 处理完成！输出已保存至: {output_txt_path} ---")

# ==============================================================================
# 6. 批量处理主程序
# ==============================================================================
if __name__ == '__main__':
    print("\n>>> [DEBUG] 步骤 6: 开始执行批量处理主程序...")

    # 加载基础配置
    if not os.path.exists(YAML_CONFIG_PATH):
        raise FileNotFoundError(f"YAML 配置文件未找到: {YAML_CONFIG_PATH}")
    with open(YAML_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # --------------------------------------------------------------------------
    # 参数搜索空间配置 (这里保持你原来的配置不变)
    # --------------------------------------------------------------------------
    param_search_space = {
        # --- 核心 ByteTrack 阈值 ---
        'track_high_thresh':   {'enable': False, 'start': 0.4, 'end': 0.6, 'step': 0.1},
        'track_low_thresh':    {'enable': False, 'start': 0.05, 'end': 0.15, 'step': 0.05},
        'new_track_thresh':    {'enable': False, 'start': 0.5, 'end': 0.7, 'step': 0.1},
        
        # --- 跟踪生命周期 ---
        'track_buffer':        {'enable': False, 'start': 60, 'end': 180, 'step': 60},
        
        # --- 关联 (IoU / Score) ---
        'match_thresh':        {'enable': False, 'start': 0.7, 'end': 0.9, 'step': 0.1},
        'second_match_thresh': {'enable': False, 'start': 0.4, 'end': 0.6, 'step': 0.1},
        'third_match_thresh':  {'enable': False, 'start': 0.6, 'end': 0.8, 'step': 0.1},
        
        # --- 深度关联 (Mahalanobis) ---
        'motion_maha_thresh':  {'enable': False, 'start': 5.991, 'end': 9.488, 'step': 1.0},
        'maha_thresh':         {'enable': False, 'start': 0.5, 'end': 30.5, 'step': 15.0},
        'depth_gate_factor':   {'enable': False, 'start': 2.0, 'end': 4.0, 'step': 0.5},
        
        # --- 深度卡尔曼滤波器 (1D) ---
        'depth_kf_R':          {'enable': False, 'start': 3.0, 'end': 7.0, 'step': 1.0},
        'depth_kf_Q_pos':      {'enable': False, 'start': 0.05, 'end': 0.15, 'step': 0.05},
        'depth_kf_Q_vel':      {'enable': False, 'start': 0.005, 'end': 0.015, 'step': 0.005},
        
        # --- 深度 ROI ---
        'depth_roi_scale':     {'enable': False,  'start': 0.15, 'end': 0.35, 'step': 0.05},
        
        # --- 布尔值 ---
        'fuse_score':          {'enable': False, 'values': [True, False]},
        'mot20':               {'enable': False, 'values': [False]}
    }

    # 1. 生成参数网格
    param_grid = {}
    enabled_params = []
    
    def get_step_values(start, end, step):
        precision = abs(int(math.log10(step))) + 1 if isinstance(step, float) and step != 0 else 0
        vals = []
        current = start
        while current <= end:
            vals.append(round(current, precision))
            current += step
            current = round(current, precision)
        if all(isinstance(v, (int, float)) and v == int(v) for v in [start, end, step]):
             return [int(v) for v in vals]
        return vals

    for param_name, config in param_search_space.items():
        if param_name not in base_config and 'values' not in config:
            continue
        if config['enable']:
            enabled_params.append(param_name)
            if 'values' in config:
                param_grid[param_name] = config['values']
            elif 'start' in config:
                param_grid[param_name] = get_step_values(config['start'], config['end'], config['step'])
        else:
            default_val = base_config.get(param_name)
            param_grid[param_name] = [default_val]

    # 2. 生成组合
    grid_keys = param_grid.keys()
    grid_value_lists = param_grid.values()
    param_combinations = [dict(zip(grid_keys, combo)) for combo in itertools.product(*grid_value_lists)]
    
    total_experiments = len(param_combinations)
    print(f"\n>>> [INFO] 总共将执行 {total_experiments} 组实验")

    # 3. 执行实验
    for idx, combo in enumerate(param_combinations, 1):
        print(f"\n" + "="*80)
        print(f">>> [INFO] 实验 {idx}/{total_experiments} - 当前参数组合:")
        current_config = combo
        
        combo_desc = []
        if enabled_params:
            for param_name in enabled_params:
                param_value = current_config[param_name]
                formatted_value = str(param_value).replace('.', 'p').replace(' ', '')
                combo_desc.append(f"{param_name}_{formatted_value}")
            output_dir_name = '_'.join(combo_desc)
        else:
            output_dir_name = 'default_params'
        
        OUTPUT_EVAL_DIR = os.path.join(BASE_OUTPUT_EVAL_DIR, output_dir_name)
        os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
        print(f">>> [INFO] 输出目录: {OUTPUT_EVAL_DIR}")
        
        if enabled_params:
            for param_name in enabled_params:
                print(f"  - [动态] {param_name}: {current_config[param_name]}")
        
        current_tracker_args = SimpleNamespace(**current_config)

        # 查找视频
        video_files = glob.glob(os.path.join(INPUT_VIDEOS_DIR, '*.mp4'))
        if not video_files:
            print(f"!!! [WARNING] 在目录 {INPUT_VIDEOS_DIR} 中未找到任何 .mp4 视频文件。")
            continue
        else:
            print(f">>> [INFO] 找到 {len(video_files)} 个视频文件进行处理。")

        # 处理每个视频
        for video_path in sorted(video_files):
            try:
                output_name = os.path.splitext(os.path.basename(video_path))[0] + '.txt'
                output_path = os.path.join(OUTPUT_EVAL_DIR, output_name)

                # ★★★ 调用时传入 RRC_DETECTIONS_DIR ★★★
                process_video_for_eval(video_path, output_path, current_tracker_args, RRC_DETECTIONS_DIR)

            except Exception as e:
                print(f"!!! [FATAL ERROR] 处理视频 {video_path} 时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n>>> [DEBUG] 所有实验处理完毕。\n" + "=" * 60)