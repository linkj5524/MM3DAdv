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
import argparse  # 导入argparse库



# 1. 创建参数解析器
parser = argparse.ArgumentParser(description="Adversarial Attack Main Program")  # 程序描述

# 2. 添加命令行参数



parser.add_argument('--exp_config_path', type=str, 
                    default="./configs/exp_config/exp_params_val.yaml",
                      help="exp config path")
parser.add_argument('--detect_config_path', type=str, 
                    default="./configs/detect_config/detect_config.yaml",
                      help="detection config path")

# 3. 解析命令行参数
args = parser.parse_args()


if __name__ == '__main__':
    # --------------------------
    # 1. 基础配置 
    # -------------------------- 



    exp_params=load_yaml_config(args.exp_config_path)
    detect_params=load_yaml_config(args.detect_config_path)

    exp_root=exp_params["experiment_path"]
    # 创建实验目录
    os.makedirs(exp_root, exist_ok=True)




    
    attack=MM3DAdv_ATTACK( 
                  exp_params=exp_params,

                  detect_params=detect_params)



    texture_path=r"exp/260112_optim_test/texture/texture.pt"
    attack.validate_adversarial_texture(texture_pt_path= texture_path,
                                        save_visual=True)

