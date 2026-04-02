import shutil
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from share import *

import numpy as np
from PIL import Image
import cv2
import einops
import numpy as np
import torch
import random
import yaml
import os
from tqdm import tqdm
import sys
from pytorch_lightning import seed_everything
import time

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from adv_attack import *
from adv_attack.attack_class import *
from adv_attack.util import *

def get_contour_canny_tensor(
    input_image: np.ndarray,
    image_resolution: int = 512,
    num_samples: int = 1,
    low_threshold: int = 50,  # 二值化/ Canny 低阈值
    high_threshold: int = 150, # 二值化/ Canny 高阈值
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    先提取图像轮廓 → 再对轮廓图执行 Canny 边缘检测，返回 B×C×H×W 格式的 float32 张量
    参数：
        input_image: 输入图像（np.ndarray），支持格式：
                     - RGB/BGR: (H, W, 3)
                     - 灰度图: (H, W)
                     - RGBA: (H, W, 4)（自动丢弃 Alpha 通道）
        image_resolution: 图像缩放分辨率（默认 512）
        num_samples: 批量维度（B，默认 1）
        low_threshold: 二值化+ Canny 低阈值（0~255，默认 50）
        high_threshold: 二值化+ Canny 高阈值（0~255，默认 150）
        device: 输出张量设备（默认自动检测 GPU/CPU）
    返回：
        torch.Tensor: 形状 (B, 3, H, W)，数值范围 0.0~1.0（0=背景，1=轮廓+Canny边缘），dtype=float32
    """
    # 1. 自动检测设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 图像预处理：统一格式+缩放
    img = HWC3(input_image)  # 转为 (H, W, 3)
    img = resize_image(img, image_resolution)  # 缩放到目标分辨率
    
    # 3. 第一步：提取图像轮廓
    # 3.1 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 3.2 二值化（用传入的阈值）
    _, binary = cv2.threshold(gray, low_threshold, high_threshold, cv2.THRESH_BINARY)
    # 3.3 提取最外层轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 3.4 绘制轮廓到空白画布
    contour_map = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(contour_map, contours, -1, (255), thickness=2)  # 轮廓线宽2
    
    # 4. 第二步：对轮廓图执行 Canny 边缘检测
    canny_on_contour = cv2.Canny(contour_map, low_threshold, high_threshold)
    # 转为3通道（保持格式统一）
    canny_on_contour = HWC3(canny_on_contour)
    
    # 5. 张量转换+归一化（0~255 → 0.0~1.0）
    control = torch.from_numpy(canny_on_contour.copy()).float().to(device) / 255.0
    
    # 6. 增加批量维度（复制num_samples份）
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    
    # 7. 调整维度顺序：B×H×W×C → B×C×H×W（PyTorch标准格式）
    control = einops.rearrange(control, 'b h w c -> b c h w').contiguous()
    
    # 8. 确保类型为float32（兼容训练）
    control = control.to(dtype=torch.float32)
    
    return control



def get_canny_edge_tensor(
    input_image: np.ndarray,
    image_resolution: int = 512,
    num_samples: int = 1,
    low_threshold: int = 50,
    high_threshold: int = 150,
    device: torch.device = None
) -> torch.Tensor:
    """
    先缩放到正方形 → 再做Canny边缘检测
    输出：B×C×H×W float32 tensor [0~1]
    """
    apply_canny = CannyDetector()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================== 最优先：先缩放到正方形 ======================
    target_size = (image_resolution, image_resolution)
    H, W = input_image.shape[:2]
    interpolation = cv2.INTER_LANCZOS4 if image_resolution > max(H, W) else cv2.INTER_AREA
    img = cv2.resize(input_image, target_size, interpolation=interpolation)
    # ==================================================================

    # 统一转 3 通道（缩放后再转）
    img = HWC3(img)

    # Canny 边缘检测
    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)

    # 转张量
    control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').contiguous()
    control = control.to(dtype=torch.float32)

    return control


def get_canny_edge_tensor1(
    input_image: np.ndarray,
    resize_size: int = 128,        # 先缩放到 512/4=128
    target_size: int = 512,        # 最终拼接为512×512
    num_samples: int = 1,          # 批量数
    low_threshold: int = 50,
    high_threshold: int = 150,
    device: Optional[torch.device] = None  # 对齐第一个函数的设备参数
) -> torch.Tensor:
    """
    图像预处理：缩放→拼接重复→Canny边缘检测，返回与get_canny_edge_tensor格式一致的张量
    核心：将128×128的图像拼接成512×512（4×4重复），非填充
    返回：
        torch.Tensor: 形状 (B, 3, 512, 512)，数值范围 0.0~1.0，dtype=float32
    """
    apply_canny = CannyDetector()
    # 1. 自动检测设备（对齐第一个函数）
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 图像预处理：强制转为 3 通道（兼容灰度/RGBA）
    img = HWC3(input_image)  # 转为 (H, W, 3)
    
    # 3. 强制缩放到 128×128（固定尺寸，便于后续拼接）
    img_resized = cv2.resize(
        img, 
        (resize_size, resize_size),  # (W, H) → 128×128
        interpolation=cv2.INTER_LANCZOS4  # 高质量缩放
    )
    
    # 4. 拼接重复为512×512（4×4网格重复128×128的图像）
    img_tiled = np.tile(img_resized, (4, 4, 1))  # (128*4, 128*4, 3) = (512,512,3)
    
    # 5. 执行 Canny 边缘检测（核心，对齐第一个函数逻辑）
    detected_map = apply_canny(img_tiled, low_threshold, high_threshold)  # (512,512) 单通道
    detected_map = HWC3(detected_map)  # 转为 (512,512,3)（3通道值相同）
    
    # 6. 转为张量 + 归一化（0~255 → 0.0~1.0）
    control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
    
    # 7. 增加批量维度（B）：复制 num_samples 份（对齐第一个函数）
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    
    # 8. 调整维度顺序：B×H×W×C → B×C×H×W（PyTorch 标准格式）
    control = einops.rearrange(control, 'b h w c -> b c h w').contiguous()
    
    # 9. 确保类型为 float32（对齐第一个函数）
    control = control.to(dtype=torch.float32)
    
    return control

def get_contour_tensor(
    input_image: np.ndarray,
    image_resolution: int = 512,
    num_samples: int = 1,
    low_threshold: int = 50,  # 二值化低阈值（适配原参数名）
    high_threshold: int = 150, # 二值化高阈值（适配原参数名）
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    对输入图像执行轮廓提取，返回 B×C×H×W 格式的 float32 张量
    参数：
        input_image: 输入图像（np.ndarray），支持格式：
                     - RGB/BGR: (H, W, 3)
                     - 灰度图: (H, W)
                     - RGBA: (H, W, 4)（自动丢弃 Alpha 通道）
        image_resolution: 图像缩放分辨率（默认 512）
        num_samples: 批量维度（B，默认 1）
        low_threshold: 二值化低阈值（0~255，默认 50）
        high_threshold: 二值化高阈值（0~255，默认 150）
        device: 输出张量设备（默认自动检测 GPU/CPU）
    返回：
        torch.Tensor: 形状 (B, 3, H, W)，数值范围 0.0~1.0（0=背景，1=轮廓），dtype=float32
    """
    # 1. 自动检测设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 图像预处理：强制转为 3 通道 + 缩放到指定分辨率
    img = HWC3(input_image)  # 转为 (H, W, 3)（兼容灰度/RGBA）
    img = resize_image(img, image_resolution)  # 缩放至目标分辨率
    
    # 3. 轮廓提取核心逻辑
    # 3.1 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 3.2 二值化（使用传入的阈值参数）
    _, binary = cv2.threshold(gray, low_threshold, high_threshold, cv2.THRESH_BINARY)
    # 3.3 提取轮廓（只保留最外层轮廓）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 3.4 创建空白画布绘制轮廓
    contour_map = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(contour_map, contours, -1, (255), thickness=2)  # thickness=2 控制轮廓线宽
    # 3.5 转为3通道（保持与Canny函数输出格式一致）
    contour_map = HWC3(contour_map)
    
    # 4. 转为张量 + 归一化（0~255 → 0.0~1.0）
    control = torch.from_numpy(contour_map.copy()).float().to(device) / 255.0
    
    # 5. 增加批量维度（B）：复制 num_samples 份
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    
    # 6. 调整维度顺序：B×H×W×C → B×C×H×W（PyTorch 标准格式）
    control = einops.rearrange(control, 'b h w c -> b c h w').contiguous()
    
    # 7. 确保类型为 float32（避免 bfloat16 等兼容问题）
    control = control.to(dtype=torch.float32)
    
    return control
import argparse  # 导入argparse库

# 1. 创建参数解析器
parser = argparse.ArgumentParser(description="Adversarial Attack Main Program")  # 程序描述

# 2. 添加命令行参数

parser.add_argument('--model_config_path', type=str, 
                    default="./configs/model_config/models_params.yaml",
                      help="model config path")

parser.add_argument('--exp_config_path', type=str, 
                    default="./configs/exp_config/exp_params.yaml",
                      help="exp config path")
parser.add_argument('--detect_config_path', type=str, 
                    default="./configs/detect_config/detect_config.yaml",
                      help="detection config path")
parser.add_argument('--adv_config_path', type=str, 
                    default="./configs/adv_config/adv_params.yaml",
                      help="adv config path")

# 3. 解析命令行参数
args = parser.parse_args()


if __name__ == '__main__':
    # --------------------------
    # 1. 基础配置 
    # -------------------------- 


    model_params=load_yaml_config(args.model_config_path)
    exp_params=load_yaml_config(args.exp_config_path)
    detect_params=load_yaml_config(args.detect_config_path)
    adv_params=load_yaml_config(args.adv_config_path)

    exp_root=exp_params["experiment_path"]
    # 创建实验目录
    os.makedirs(exp_root, exist_ok=True)
    # 将yaml文件复制到实验目录
    shutil.copy(args.model_config_path, exp_root)
    shutil.copy(args.exp_config_path, exp_root)
    shutil.copy(args.detect_config_path, exp_root)
    shutil.copy(args.adv_config_path, exp_root)



    
    attack=MM3DAdv_ATTACK( 
                  model_params=model_params,
                  exp_params=exp_params,
                  adv_params=adv_params,
                  detect_params=detect_params)

    imgsize_width=attack.exp_params["image_size"]

    img = cv2.imread(r'./test_imgs/texture.jpg')  # BGR 格式 (H, W, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
    
    # 2. 调用 Canny 函数
    canny_tensor = get_canny_edge_tensor(
        input_image=img,
        image_resolution=512,
        num_samples=1,
        low_threshold=100,
        high_threshold=200
    )

    # canny_tensor = get_contour_tensor(
    #     input_image=img,
    #     image_resolution=512,
    #     num_samples=1,
    #     low_threshold=10,
    #     high_threshold=200
    # )
    
    # canny_tensor = get_contour_canny_tensor(
    #     input_image=img,
    #     image_resolution=512,
    #     num_samples=1,
    #     low_threshold=10,
    #     high_threshold=200 
    # )

    # canny_tensor = get_canny_edge_tensor1(
    #     input_image=img,
    #     resize_size=128,       # 512/4=128
    #     target_size=512,       # 最终拼接为512×512
    #     num_samples=1,
    #     low_threshold=50,
    #     high_threshold=150
    # )

    if canny_tensor.dim()==3:  # 添加维度
        canny_tensor = canny_tensor.unsqueeze(0)
    
    attack.generate_adversarial_com(canny_tensor)

