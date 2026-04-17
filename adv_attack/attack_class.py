
import json
import time

from omegaconf import OmegaConf

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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
# from torch.amp import autocast, GradScaler 
from torch.cuda.amp import autocast, GradScaler 
# 本地的包
## 添加本地包路径,即上一级的路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import instantiate_from_config as instantiate_from_config_vae
# from util import *

# 
from adv_attack.util import *

##
sys.path.append(os.path.dirname(__file__))
from object_detection_class import *
from sam import *
from captioner_blip_model import *
from sd_inpaint import *
from IDG_util.attribution_methods.saliencyMethods import * 
from vae import *
from ADVLogo_attack_tools import *
from fgsm_attack_tools import *
from pt3D_tools import *



class MM3DAdv_ATTACK:
    def __init__(self,
                  model_params:dict=None,
                  exp_params:dict=None,
                  adv_params:dict=None,
                  detect_params:dict=None
                  ):
        """
        初始化对抗攻击类
        
        参数:
            device: 运行设备 (默认 "cuda")
            exp_params: 实验参数 (默认 None),包含训练轮数、学习率,权重,bf,exp 路径
            adv_params: 对抗攻击参数 (默认 None)
            detect_params: 目标检测参数 (默认 None)
        """
    
        # 加载模型配置
        self.model_params = model_params
        self.detect_params=detect_params
        self.exp_params=exp_params
        self.adv_params=adv_params

        self.class_names_ymal=detect_params['nclass_yaml_path']
        self.optim = type('OptimContainer', (), {})()  # 动态创建空对象


    # 初始化controlnet模型
    def init_controlnet(self):
        """初始化ControlNet模型"""
                # 初始化模型
        config_path=self.model_params['model_types']['controlnet']
        model_path=self.model_params['model_paths']['controlnet']
        self.model = create_model(config_path).cpu()
        self.model.load_state_dict(load_state_dict(model_path, location='cuda'),strict=False)
        self.ddim_sampler = DDIMSampler(self.model)


    # 模型destroy
    def destroy_controlnet(self):
        """销毁模型"""
        # 判断模型是否已经初始化
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ddim_sampler'):
            del self.ddim_sampler

        torch.cuda.empty_cache()




    def init_object_detection(self,device=None):
        """初始化目标检测模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载检测模型


        self.object_detection=ObjectDetection(device=device,
                                              **self.detect_params)
        
        model_type=self.detect_params['attack_model']["model_type"]
        modelt_path=self.detect_params['attack_model']["model_path"]
        self.object_detection.load_model( model_type=model_type,
                                    model_path=modelt_path
                                    )
        
    def init_object_detection_return(self,device=None):
        detect_params=self.detect_params
        """初始化目标检测模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载检测模型


        object_detection=ObjectDetection(device=device,
                                            **detect_params)
        
        model_type=detect_params['attack_model']["model_type"]
        modelt_path=detect_params['attack_model']["model_path"]
        object_detection.load_model( model_type=model_type,
                                    model_path=modelt_path
                                    )
        return object_detection

    def destroy_object_detection(self,object_detection):
        """销毁目标检测模型"""
        if object_detection is not None:
            del object_detection
        # 清空内存
        torch.cuda.empty_cache()


    def detect_val(self,input_image,
                   input_path,
                   input_file_name,
                   ):
        """初始化目标检测模型"""
        detect_params=self.detect_params
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载检测模型
        object_detection_val=ObjectDetection(device=device,
                                              ** detect_params)
        detect_result_list={}
        for val_model_key in detect_params['val_model1'].keys():
            model_type_val=detect_params['val_model1'][val_model_key]["model_type"]
            model_path_val=detect_params['val_model1'][val_model_key]["model_path"]
            if len(model_path_val)<=0 :
                object_detection_val.load_model( model_type=model_type_val)
            else:                        
                object_detection_val.load_model( model_type=model_type_val,
                                        model_path=model_path_val
                                        )
            # 检测
            image_name=input_file_name+model_type_val+'.jpg'
            temp_result,_=object_detection_val.detect_eval(images=input_image,
                                                model_type=model_type_val,
                                                file_path=input_path,
                                                file_name=image_name)
            # 删除模型
            if    model_type_val in object_detection_val.models:                     
                    del object_detection_val.models[model_type_val]
            detect_result_list[model_type_val]=temp_result
            
            # 清空内存
            torch.cuda.empty_cache()
        return detect_result_list
            
        



    # 初始纹理生成
    def init_tex_generate(self,control_image=None, 
                            ref_class=None,
                            negtive_class=None,
                            cam_target_class=None,
                            ):


        
        """
            ====================================================
            =========== controlnet 的初始化,采样 ===============
            ====================================================
        """

        B,C,H, W= control_image.shape
        shape = (4, H // 8, W // 8)

        


        # 初始化模型
        self.init_controlnet()
        # 条件编码部分 放在GPU
        if self.adv_params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)


        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

   



        """
            ====================================================
            =========== controlnet采样 ===============
            ====================================================
        """
        # control_text=[s1+" . "+s2+" . "+s1+params["prompt"] for s1,s2 in   zip(object_class,object_imag_caption)]
        if ref_class is not None:
            control_text=[' '.join([s1]*1)+" . "+" . "+s1+self.adv_params["prompt"] for s1 in   ref_class] # 目前较正常
        elif cam_target_class is not None:
            control_text=[' '.join([s1]*1)+" . "+" . "+s1+self.adv_params["prompt"] for s1 in   cam_target_class]
        else :
            control_text=[self.adv_params["prompt"] ]*B

        # if negtive_class is not None:
        #     negtive_control_text=[' '.join([s1]*5)+" . "+" . "+s1+self.adv_params["n_prompt"] for s1 in   negtive_class]
        # else :
        #     negtive_control_text=[self.adv_params["n_prompt"]] * B

        if negtive_class is not None:
            # 有负面类别：类别 + 负面提示，然后复制 B 份（匹配 batch）
            negtive_control_text = [negtive_class + self.adv_params["n_prompt"]] * B
        else:
            # 没有：只用默认负面提示，复制 B 份
            negtive_control_text = [self.adv_params["n_prompt"]] * B
            
        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    control_text  
                )
            ]
        }
        un_cond = {
            "c_concat": None if self.adv_params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                     negtive_control_text   # [params["n_prompt"]] * B
                )
            ]
        }
 
        self.model.control_scales = (
            [self.adv_params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if self.adv_params["guess_mode"]
            else [self.adv_params["strength"]] * 13
        ) 
        # 切换扩散部分放在GPU
        if self.adv_params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)
        st_time=time.time()
        with torch.no_grad():
            samples, intermediates = self.ddim_sampler.sample(self.adv_params["ddim_steps"], B,
                                                    shape, cond, verbose=False, eta=self.adv_params["eta"],
                                                    unconditional_guidance_scale=self.adv_params["scale"],
                                                    unconditional_conditioning=un_cond)            
        
        
            controlnet_adv_sample = self.model.decode_first_stage(samples)
        end = time.time()
        print(f"sample time:{end-st_time:.2f}")        
        self.destroy_controlnet() 


        controlnet_adv_sample=(controlnet_adv_sample+1)/2 # 采样原始范围为-1到1，这里转为0-1
        # 限制范围
        controlnet_adv_sample=torch.clamp(controlnet_adv_sample,0,1)

        # 返回0-1的tensor
        return controlnet_adv_sample


    def to_imgTensor_from_numpy_int8(self, image):
        """
        作用：将numpy数组转换为PyTorch张量。
        参数：
        image: 输入的numpy数组，形状为[C, H, W]。
        返回：
        tensor: 转换后的PyTorch张量，形状为[C, H, W]。
        """

        # 转换到-1到1
        image = image.astype(np.float32) / 127.5 - 1.0

        tensor = torch.from_numpy(image).float()

        
        return tensor.unsqueeze(0)
    

    # 将图片转化为latent
    def imgTensor_to_latent(self, img,scale=0.18215):
        '''
        img:[-1-1],type:tensor
        return: latent, type:tensor
        '''
        
        #编码为潜变量（关闭梯度计算，提高效率）
        with torch.no_grad():
            posterior = self.model.first_stage_model.encode(img)  # 得到后验分布
            
            # # 4. 从分布中获取潜变量
            # if sample_posterior:
            #     z = posterior.sample()  # 随机采样（带随机性）
            # else:
            z = posterior.mode()    # 取均值（确定性结果，推荐用于推理）
        z=z*scale
        z=z.to(self.device)
        return z   
    # def latent_to_imgTensor01(self,latent):
    #     img = self.model.first_stage_model.decode(latent)
    #     return  (einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    def latent_to_imgTensor01(self,latent,scale=0.18215):
        latent = latent / scale
        img = self.model.first_stage_model.decode(latent)
        # sd 对应的区间为-1到1，需要转换到0到1
        img = ((img + 1)*0.5 ).to(dtype=torch.float32)
    
        # # 确保与YOLO模型在同一设备
        # img = img.to(self.yolo_model.device)  # 假设self.yolo_model是加载的YOLO模型
        
        return img
        # return  (einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)


    def tensor_01_to_numpy_255(self,tensor):
        """
        将模型输出的0-1范围图像张量转换为0-255范围numpy数组，并调整通道顺序
        
        Args:
            tensor: 模型输出的图像张量，格式为 [B, C, H, W] 或 [C, H, W]（单张图像）
                    数值范围必须是 [0, 1]，通道数通常为1（灰度）或3（RGB/BGR）
            is_rgb: 若为True，默认输入通道为RGB（无需额外转换）；
                    若为False，会将RGB转为BGR（适配opencv的默认通道顺序）
        
        Returns:
            numpy_array: 转换后的numpy数组，格式为 [B, H, W, C] 或 [H, W, C]（单张图像）
                        数值范围 [0, 255]，数据类型 uint8
        """
        # -------------------------- 1. 处理单张图像（无批量维度） --------------------------
        if tensor.dim() == 3:  # 输入为 [C, H, W]（单张图像），添加批量维度变为 [1, C, H, W]
            tensor = tensor.unsqueeze(0)
        

        # -------------------------- 2. 设备迁移 + 张量转numpy --------------------------
        # 推理阶段用 .detach() 切断梯度，训练阶段若需保留梯度可移除（但通常图像转换用于推理）
        if tensor.is_cuda:
            tensor = tensor.cpu()  # 移到CPU（numpy不支持CUDA数据）
        np_array = tensor.detach().numpy()  # 张量 → numpy数组，格式 [B, C, H, W]

        # -------------------------- 3. 0-1 → 0-255 缩放 + 数据类型转换 --------------------------
        # 乘以255后用np.clip确保数值在0-255（避免浮点误差导致的超界，如1.0001→255.025）
        np_array = np.clip(np_array * 255.0, a_min=0, a_max=255)

        np_array = np_array.astype(np.uint8)

        np_array = np.transpose(np_array, axes=(0, 2, 3, 1))  # 调整维度顺序

 

        if np_array.shape[0] == 1:
            np_array = np_array.squeeze(0)  # 从 [1, H, W, C] 变为 [H, W, C]

        return np_array



    def optim_prepare(self,
        device=None,
        ini_texture=None,
        data_type=None):
        # ========== 1. 新增AMP/BF16参数解析 ==========
        use_amp = self.exp_params.get("use_amp", False)  # 是否启用AMP
        use_bf16 = self.exp_params.get("use_bf16", False)  # 是否启用BF16（优先级高于FP32）
        assert not (use_amp and use_bf16), "AMP和BF16不能同时启用"

        # ========== 2. 设备和精度初始化 ==========
        if device is None:
             optim_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            optim_device = device

        # 精度适配：BF16 / FP32
        if use_bf16 and torch.cuda.is_bf16_supported():
            optim_data_type = torch.bfloat16
        elif data_type is not None:
            optim_data_type = data_type
        else:
            optim_data_type = torch.float32

        # ========== 3. AMP GradScaler初始化（仅AMP模式） ==========
        scaler = GradScaler() if use_amp else None

        # ========== 原有逻辑：数据预处理 ==========

        


        # 数据移至设备并转换精度
        adv_init_tensor = move_to_gpu_and_cast_dtype(ini_texture, optim_device, optim_data_type)
        adv_init_tensor_gt = adv_init_tensor.clone()


        # VAE初始化
        vae_optim = None
        if self.exp_params["optim_object_type"] != 1:
            vae_model_path=self.model_params['model_paths']['vae_model']
            vae_optim = VAEInferencer(model_name=vae_model_path, dtype=optim_data_type)

        # 优化器初始化
        if self.exp_params["optim_object_type"] == 0:
            # VAE latent优化
            adv_init_latent = vae_optim.encode_infer(adv_init_tensor * 2 - 1)
            adv_init_latent = move_to_gpu_and_cast_dtype(adv_init_latent, optim_device, optim_data_type)
            adv_init_latent = adv_init_latent.clone().detach()
            adv_init_latent.requires_grad = True
            optimizer = torch.optim.Adam([adv_init_latent], lr=self.exp_params["lr"])
        else:
            # RGB优化
            adv_init_tensor.requires_grad = True
            optimizer = torch.optim.Adam([adv_init_tensor], lr=self.exp_params["lr"])

        # 学习率调度器
        scheduler = StepLR(
            optimizer,
            step_size=self.exp_params["lr_step"],
            gamma=self.exp_params["lr_decay"]
        )

        # 损失函数初始化（适配精度）
        cross_entro_loss = YOLOv11DetectionLoss(**self.detect_params, **self.exp_params).to(optim_device, dtype=optim_data_type)
        huvloss=HistogramUVLoss()
        TV_Loss = TVLoss().to(optim_device, dtype=optim_data_type) if self.exp_params["TV_loss_weight"] > 0 else None
        conext_loss_l2 = MaskedL1L2Loss().to(optim_device, dtype=optim_data_type) if self.exp_params["conext_loss_weight"] > 0 else None
        perceptual_loss = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(optim_device, dtype=optim_data_type) if self.exp_params["perceptual_loss_weight"] > 0 else None

        detect_model_type = self.detect_params["attack_model"]["model_type"]

        # pr_scale=torch.tensor(mask_pre[0])
        # pr_scale= 1 #1/pr_scale # 单位像素感知损失差距，所以需要除以比例

        # data_load 初始化
        transforms_list = transforms.Compose([
                    transforms.ToTensor()  # 转为张量 [C, H, W]，值归一化到 [0,1]
                    ]) 
        train_loader ,val_loader=build_from_cfg(self.exp_params["dataloader"],
                                             batch_size=self.exp_params["batch_size"],
                                              num_workers=self.exp_params["num_workers"],)


        # train_loader ,val_loader=build_RGBCameraPose_dataloader(
        #         train_root_dir=self.exp_params["train_root_dir"],
        #         val_root_dir=self.exp_params["val_root_dir"],
        #         batch_size=self.exp_params["batch_size"],
        #         num_workers=self.exp_params["num_workers"],
        #         transform= transforms_list)
        
        self.optim.object_detect = self.init_object_detection_return(
            device=optim_device)

        self.optim.optim_device= optim_device
        self.optim.optim_data_type= optim_data_type
        # loss
        self.optim.cross_entro_loss= cross_entro_loss
        self.optim.TV_Loss= TV_Loss
        self.optim.conext_loss_l2= conext_loss_l2
        self.optim.perceptual_loss= perceptual_loss
        self.optim.huvloss=huvloss
        # related model
        self.optim.vae_optim= vae_optim
        self.optim.detect_model_type= detect_model_type
        # optimizer and scheduler
        self.optim.optimizer= optimizer
        self.optim.scheduler= scheduler
        # scaler
        self.optim_scaler= scaler

        # related tensor
        self.optim.adv_init_tensor_gt= adv_init_tensor_gt
        self.optim.adv_init_tensor= adv_init_tensor
        self.optim.adv_init_latent= adv_init_latent

        # loader
        self.optim.train_loader= train_loader
        self.optim.val_loader= val_loader
        
        mesh_model_t,material_list=load_obj_model_return_mesh_material(self.exp_params["mesh_model_path"], 
                                optim_device)
        self.optim.mesh_model =mesh_model_t
        self.optim.material_list=material_list
        self.optim.target_material=self.exp_params["target_material_dict"]


    # def optim_step(self):

    #     epoch_false_pos_rates = []
    #     self.optim.optimizer.zero_grad()
    #     use_amp = self.exp_params.get("use_amp", False)  # 是否启用AMP
    #     use_bf16 = self.exp_params.get("use_bf16", False)  # 是否启用BF16（优先级高于FP32）
    #     for epoch in tqdm(range(self.exp_params["optim_epochs"]), desc="Optimizing"):
    #         epoch_total_fp = 0
    #         epoch_batch_count = 0
    #         step_num=0
    #         for backgroud_images ,cameras_pose_path in self.optim.train_loader:
    #             backgroud_images=backgroud_images.to(self.optim.optim_device)
    #             # ========== 前向传播（AMP上下文） ==========
    #             with autocast(enabled=use_amp,
    #                             dtype=torch.bfloat16 if use_bf16 else torch.float16):
    #                 # 生成对抗样本
    #                 if self.exp_params ["optim_object_type"] == 0:
    #                     adv_tensor_generate01 = self.optim.vae_optim.decode_infer(self.optim.adv_init_latent)
    #                     adv_tensor_generate = (adv_tensor_generate01 + 1) / 2
    #                 elif self.exp_params["optim_object_type"] == 1:
    #                     adv_tensor_generate = self.optim.adv_init_tensor
    #                 elif self.exp_params["optim_object_type"] == 2:
    #                     adv_init_tensor1 = self.optim.adv_init_tensor * 2 - 1
    #                     adv_tensor_generate01 = self.optim.vae_optim.infer(adv_init_tensor1, sample_posterior=False)
    #                     adv_tensor_generate = (adv_tensor_generate01 + 1) / 2



    #                 # resize
    #                 image_size_render=( self.exp_params["render_size"]["height"], 
    #                             self.exp_params["render_size"]["width"])
    #                 # resize
    #                 # adv_texture_resized = resize_tensor(adv_tensor_generate, 
    #                 #                                 height=image_size_render[0], 
    #                 #                                     width=image_size_render[1])
    #                 # render 渲染
    #                 # 初始化object mesh

    #                 ## 直接render
                    
    #                 origin_com_tensor_rendered,object_mask = load_parma_and_render_main(object_mesh=self.optim.mesh_model,
    #                                                                 background=backgroud_images,
    #                                                                 path_camera_pose=cameras_pose_path,
    #                                                                 image_size=image_size_render,
    #                                                                 device=self.optim.optim_device,
    #                                                                 fov=110,
    #                                                                 blur_radius=0.0,
    #                                                                 faces_per_pixel=1)
    #                 ## 纹理贴图渲染
    #                 if self.optim.target_material is None or  len(self.optim.target_material) != len(adv_tensor_generate)   :


    #                     # 如果不对应，默认全部渲染
    #                     adv_tex = adv_tensor_generate[0] if adv_tensor_generate.ndim == 4 else adv_tensor_generate
    #                     target_index_dict = {mat: adv_tex for mat in self.optim.material_list}

    #                 else:
                        
    #                     target_index_dict = {}
    #                     for mat, tex in zip(self.optim.target_material, adv_tensor_generate):
    #                         target_index_dict[mat] = tex


    #                 new_mesh_rendered_adv_com=update_meshes_texture_dict(
    #                     original_meshes_list=self.optim.mesh_model,
    #                     target_index_dict=target_index_dict,           # 形状为 [1, C, H, W] 的纹理张量
    #                     material_names_list=self.optim.material_list,
    #                     device=self.optim.optim_device)
    #                 adv_com_tensor_rendered,_ = load_parma_and_render_main(object_mesh=new_mesh_rendered_adv_com,
    #                                                                 background=backgroud_images,
    #                                                                 path_camera_pose=cameras_pose_path,
    #                                                                 image_size=image_size_render,
    #                                                                 device=self.optim.optim_device,
    #                                                                 fov=110,
    #                                                                 blur_radius=0.0,
    #                                                                 faces_per_pixel=1)
                    

    #                 # 目标模型检测
    #                 ## origin detection

    #                 ## adv com detection
    #                 detect_image_size=self.exp_params["image_size"]
    #                 adv_com_tensor_rendered_sized=resize_tensor_ratio_pad(adv_com_tensor_rendered, 
    #                                                 height=detect_image_size, 
    #                                                     width=detect_image_size)
                    
    #                 detect_model_type =self.detect_params["attack_model"]["model_type"]
    #                 # 可视化路径
    #                 detect_visual_path_list=[os.path.join( self.exp_params["visual_path"] , f"step_{step_num %20}_{i}") for i in range(self.exp_params["batch_size"])]
                    
    #                 result_object_adv_com, _ = self.optim.object_detect.detect_eval(
    #                     adv_com_tensor_rendered_sized,
    #                     file_path=detect_visual_path_list,
    #                     file_name='adv_detect.jpg',
    #                     grad_status=True,
    #                     model_type=detect_model_type
    #                 )

    #                 origin_com_tensor_rendered_sized=resize_tensor_ratio_pad(origin_com_tensor_rendered, 
    #                                                 height=detect_image_size, 
    #                                                     width=detect_image_size)
                    

                    
    #                 result_object_origin, _ = self.optim.object_detect.detect_eval(
    #                     origin_com_tensor_rendered_sized,# origin_com_tensor_rendered_sized,
    #                     file_path=detect_visual_path_list,
    #                     file_name='origin_detect.jpg',
    #                     grad_status=True,
    #                     model_type=detect_model_type
    #                 )

    #                 # 获取mask
    #                 object_mask_resize=resize_tensor_ratio_pad(object_mask, 
    #                                                 height=detect_image_size, 
    #                                                     width=detect_image_size)
    #                 result_object_origin_only_object_gt=mask_to_gt_dict(
    #                         mask_tensor=object_mask_resize,
    #                         label=self.exp_params['target_class'],
    #                         num_classes=self.detect_params["nums_class"],
    #                         device=self.optim.optim_device,
    #                         threshold = 1e-3
    #                     )
                    

    #                 origin_com_tensor_rendered_sized=move_to_gpu_and_cast_dtype(origin_com_tensor_rendered_sized, self.optim.optim_device, self.optim.optim_data_type)
    #                 adv_com_tensor_rendered_sized=move_to_gpu_and_cast_dtype(adv_com_tensor_rendered_sized, self.optim.optim_device, self.optim.optim_data_type)
    #                 backgroud_images=move_to_gpu_and_cast_dtype(backgroud_images, self.optim.optim_device, self.optim.optim_data_type)

    #                 # 各损失计算

    #                 tv_loss = torch.tensor(0.0, device=self.optim.optim_device, dtype=self.optim.optim_data_type)
    #                 if self.exp_params["TV_loss_weight"] > 0:
    #                     tv_loss = self.optim.TV_Loss(adv_com_tensor_rendered_sized)

    #                 conext_loss = torch.tensor(0.0, device=self.optim.optim_device, dtype=self.optim.optim_data_type)
    #                 if self.exp_params["conext_loss_weight"] > 0:
    #                     mask=torch.ones_like(adv_com_tensor_rendered_sized)
    #                     conext_loss = self.optim.conext_loss_l2(adv_com_tensor_rendered_sized, origin_com_tensor_rendered_sized, mask)

    #                 pr_loss = torch.tensor(0.0, device=self.optim.optim_device, dtype=self.optim.optim_data_type)
    #                 if  self.exp_params["perceptual_loss_weight"] > 0 :
    #                     pr_loss = self.optim.perceptual_loss(normalize_to_01(origin_com_tensor_rendered_sized),
    #                                                         normalize_to_01(adv_com_tensor_rendered_sized))

    #                 # UV 损失
    #                 huv_loss = torch.tensor(0.0, device=self.optim.optim_device, dtype=self.optim.optim_data_type)
    #                 if self.exp_params["HUV_loss_weight"] > 0:
    #                     huv_loss = self.optim.huvloss(xt=adv_tensor_generate,bg=backgroud_images)

                    

    #                 # 检测损失,默认输出对抗损失，即需要最小化loss。

    #                 loss, loss_dict = self.optim.cross_entro_loss(result_object_adv_com,result_object_origin_only_object_gt)

    #                 # 总损失
    #                 total_loss = (
    #                     self.exp_params["TV_loss_weight"] * tv_loss
    #                     + self.exp_params["perceptual_loss_weight"] * pr_loss
    #                     + self.exp_params["conext_loss_weight"] * conext_loss
    #                     + self.exp_params["class_loss_weight"]*loss_dict['class_loss']
    #                     + self.exp_params["HUV_loss_weight"]*huv_loss
    #                 )

    #             # ========== 反向传播 + 优化（AMP适配） ==========
    #             if use_amp:
    #                 # AMP模式：缩放梯度避免下溢
    #                 self.optim.scaler.scale(total_loss).backward()
    #                 self.optim.scaler.step(self.optim.optimizer)
    #                 self.optim.scaler.update()
    #             else:
    #                 # 普通模式/BF16模式
    #                 total_loss.backward()
    #                 self.optim.optimizer.step()

    #             # 学习率调度
    #             self.optim.scheduler.step()

    #             # 裁剪参数范围
    #             if self.exp_params["optim_object_type"] != 0:
    #                 self.optim.adv_init_tensor.data = torch.clamp(self.optim.adv_init_tensor.data, 0.0, 1.0)

    #             temp_dict=count_false_positive_single_model(model_result= result_object_adv_com,
    #                                                         ref_result=result_object_origin_only_object_gt,
    #                                                         iou_threshold= 0.5,# self.detect_params['iou_threshold'],
    #                                                         conf_threshold=0.5 )#self.detect_params['conf_threshold'])
    #             # print(temp_dict)
    #             epoch_total_fp += temp_dict["avg_false_pos_prob"]
    #             epoch_batch_count += sum(temp_dict['false_pos_counts'])
                

    #         if epoch_batch_count > 0:
    #             avg_fp_epoch = epoch_batch_count / (step_num+1)
    #             print(f"\n Epoch [{epoch+1}] 平均误识别率: {avg_fp_epoch:.4f}")
    #             epoch_false_pos_rates.append(avg_fp_epoch)

    

    #         # 保存 texture
    #         if step_num %self.exp_params["texture_save_interval"]==0:
    #             save_tensor_root=os.path.join(self.exp_params["experiment_path"],"texture")
    #             os.makedirs(save_tensor_root,exist_ok=True)

    #             save_tensor_path=os.path.join(save_tensor_root,"texture_{step_num}.pt") 
    #             torch.save(adv_tensor_generate, save_tensor_path)

    #     # 
    #     val_detect_path_list=[os.path.join( self.exp_params["visual_path"] , f"val_detect_{i}") for i in range(self.exp_params["batch_size"])]  
    #     ref_result_dict=self.detect_val(
    #         input_image=adv_com_tensor_rendered_sized,
    #         input_path=val_detect_path_list,
    #         input_file_name='adv_detect_val' # 
    #     )

    #     return epoch_false_pos_rates,ref_result_dict
        
    def optim_step(self):
        epoch_false_pos_rates = []
        use_amp = self.exp_params.get("use_amp", False)
        use_bf16 = self.exp_params.get("use_bf16", False)

        # 优化器梯度清空（应该放在 epoch 外）
        self.optim.optimizer.zero_grad()

        for epoch in tqdm(range(self.exp_params["optim_epochs"]), desc="Optimizing"):
            epoch_total_fp = 0.0
            epoch_batch_count = 0
            step_num = 0  # ← 放在 epoch 内才对

            for backgroud_images, cameras_pose_path in self.optim.train_loader:
                backgroud_images = backgroud_images.to(self.optim.optim_device)

                # ===================== 前向传播 =====================
                with autocast(enabled=use_amp, dtype=torch.bfloat16 if use_bf16 else torch.float16):

                    # 生成对抗纹理
                    if self.exp_params["optim_object_type"] == 0:
                        adv_tensor_generate01 = self.optim.vae_optim.decode_infer(self.optim.adv_init_latent)
                        adv_tensor_generate = (adv_tensor_generate01 + 1) / 2
                    elif self.exp_params["optim_object_type"] == 1:
                        adv_tensor_generate = self.optim.adv_init_tensor
                    elif self.exp_params["optim_object_type"] == 2:
                        adv_init_tensor1 = self.optim.adv_init_tensor * 2 - 1
                        adv_tensor_generate01 = self.optim.vae_optim.infer(adv_init_tensor1, sample_posterior=False)
                        adv_tensor_generate = (adv_tensor_generate01 + 1) / 2
                    else:
                        raise ValueError(f"不支持的 optim_object_type: {self.exp_params['optim_object_type']}")

                    # 渲染尺寸,0：h,1:w
                    image_size_render = (self.exp_params["render_size"]["height"],
                                        self.exp_params["render_size"]["width"])

                    # ===================== 原始车辆渲染 =====================
                    origin_com_tensor_rendered, object_mask = load_parma_and_render_main(
                        object_mesh=self.optim.mesh_model,
                        background=backgroud_images,
                        path_camera_pose=cameras_pose_path,
                        image_size=image_size_render,
                        device=self.optim.optim_device,
                        fov=110,
                        blur_radius=0.0,
                        faces_per_pixel=1
                    )

                    # ===================== 对抗纹理渲染 =====================
                    if self.optim.target_material is None or len(self.optim.target_material) != len(adv_tensor_generate):
                        adv_tex = adv_tensor_generate[0] if adv_tensor_generate.ndim == 4 else adv_tensor_generate
                        target_index_dict = {mat: adv_tex for mat in self.optim.material_list}
                    else:
                        target_index_dict = {}
                        for mat, tex in zip(self.optim.target_material, adv_tensor_generate):
                            target_index_dict[mat] = tex

                    # 更新纹理并渲染
                    new_mesh_rendered_adv_com = update_meshes_texture_dict(
                        original_meshes_list=self.optim.mesh_model,
                        target_index_dict=target_index_dict,
                        material_names_list=self.optim.material_list,
                        device=self.optim.optim_device
                    )

                    adv_com_tensor_rendered, _ = load_parma_and_render_main(
                        object_mesh=new_mesh_rendered_adv_com,
                        background=backgroud_images,
                        path_camera_pose=cameras_pose_path,
                        image_size=image_size_render,
                        device=self.optim.optim_device,
                        fov=110,
                        blur_radius=0.0,
                        faces_per_pixel=1
                    )

                    # ===================== 检测输入缩放 =====================
                    detect_image_size = self.exp_params["image_size"]
                    adv_com_tensor_rendered_sized = resize_tensor_ratio_pad(
                        adv_com_tensor_rendered, height=detect_image_size, width=detect_image_size
                    )
                    origin_com_tensor_rendered_sized = resize_tensor_ratio_pad(
                        origin_com_tensor_rendered, height=detect_image_size, width=detect_image_size
                    )
                    object_mask_resize = resize_tensor_ratio_pad(
                        object_mask, height=detect_image_size, width=detect_image_size
                    )

                    # ===================== 检测模型 =====================
                    detect_model_type = self.detect_params["attack_model"]["model_type"]
                    # 可视化路径
                    detect_visual_path_list = [
                        os.path.join(self.exp_params["visual_path"], f"step_{step_num % 20}_{i}")
                        for i in range(backgroud_images.shape[0])  # ← 修复动态 batch
                    ]

                    # 对抗样本检测
                    result_object_adv_com, _ = self.optim.object_detect.detect_eval(
                        adv_com_tensor_rendered_sized,
                        file_path=detect_visual_path_list,
                        file_name='adv_detect.jpg',
                        grad_status=True,
                        model_type=detect_model_type
                    )

                    # 原始样本检测
                    result_object_origin, _ = self.optim.object_detect.detect_eval(
                        origin_com_tensor_rendered_sized,
                        file_path=detect_visual_path_list,
                        file_name='origin_detect.jpg',
                        grad_status=True,
                        model_type=detect_model_type
                    )

                    # GT 从 mask 生成
                    result_object_origin_only_object_gt = mask_to_gt_dict(
                        mask_tensor=object_mask_resize,
                        label=self.exp_params['target_class'],
                        num_classes=self.detect_params["nums_class"],
                        device=self.optim.optim_device,
                        threshold=1e-3
                    )

                    # ===================== 数据类型与设备统一 =====================
                    origin_com_tensor_rendered_sized = move_to_gpu_and_cast_dtype(
                        origin_com_tensor_rendered_sized, self.optim.optim_device, self.optim.optim_data_type
                    )
                    adv_com_tensor_rendered_sized = move_to_gpu_and_cast_dtype(
                        adv_com_tensor_rendered_sized, self.optim.optim_device, self.optim.optim_data_type
                    )
                    backgroud_images = move_to_gpu_and_cast_dtype(
                        backgroud_images, self.optim.optim_device, self.optim.optim_data_type
                    )

                    # ===================== 损失计算 =====================
                    tv_loss = self.optim.TV_Loss(adv_com_tensor_rendered_sized) if self.exp_params["TV_loss_weight"] > 0 else 0.0
                    conext_loss = self.optim.conext_loss_l2(adv_com_tensor_rendered_sized, origin_com_tensor_rendered_sized, torch.ones_like(adv_com_tensor_rendered_sized)) if self.exp_params["conext_loss_weight"] > 0 else 0.0
                    pr_loss = self.optim.perceptual_loss(normalize_to_01(origin_com_tensor_rendered_sized), normalize_to_01(adv_com_tensor_rendered_sized)) if self.exp_params["perceptual_loss_weight"] > 0 else 0.0
                    huv_loss = self.optim.huvloss(xt=adv_tensor_generate, bg=backgroud_images) if self.exp_params["HUV_loss_weight"] > 0 else 0.0

                    # 检测损失
                    loss, loss_dict = self.optim.cross_entro_loss(result_object_adv_com, result_object_origin_only_object_gt)

                    # 总损失
                    total_loss = (
                        self.exp_params["TV_loss_weight"] * tv_loss
                        + self.exp_params["perceptual_loss_weight"] * pr_loss
                        + self.exp_params["conext_loss_weight"] * conext_loss
                        + self.exp_params["class_loss_weight"] * loss_dict['class_loss']
                        + self.exp_params["HUV_loss_weight"] * huv_loss
                    )

                # ===================== 反向传播 =====================
                if use_amp:
                    self.optim.scaler.scale(total_loss).backward()
                    self.optim.scaler.step(self.optim.optimizer)
                    self.optim.scaler.update()
                else:
                    total_loss.backward()
                    self.optim.optimizer.step()

                # 学习率 + 梯度清空（每步都要清空）
                self.optim.scheduler.step()
                self.optim.optimizer.zero_grad()

                # 纹理裁剪
                if self.exp_params["optim_object_type"] != 0:
                    self.optim.adv_init_tensor.data.clamp_(0.0, 1.0)

                # ===================== 误检率计算 =====================
                temp_dict = count_false_positive_single_model(
                    model_result=result_object_adv_com,
                    ref_result=result_object_origin_only_object_gt,
                    iou_threshold=0.5,
                    conf_threshold=0.5
                )

                epoch_total_fp += temp_dict["avg_false_pos_prob"]
                epoch_batch_count += sum(temp_dict['false_pos_counts'])
                step_num += 1  # ← 必须 step +1

                # ===================== 保存纹理 =====================
                if step_num % self.exp_params["texture_save_interval"] == 0:
                    save_tensor_root = os.path.join(self.exp_params["experiment_path"], "texture")
                    os.makedirs(save_tensor_root, exist_ok=True)
                    save_tensor_path = os.path.join(save_tensor_root, f"texture_{step_num}.pt")
                    torch.save(adv_tensor_generate.detach().cpu(), save_tensor_path)

            # ===================== Epoch 结果输出 =====================
            total_steps = step_num if step_num > 0 else 1
            avg_fp_epoch = epoch_batch_count / total_steps
            print(f"\n Epoch [{epoch+1}] 平均误识别数: {avg_fp_epoch:.4f}")
            epoch_false_pos_rates.append(avg_fp_epoch)

        # ===================== 最终验证 =====================
        val_detect_path_list = [
            os.path.join(self.exp_params["visual_path"], f"val_detect_{i}")
            for i in range(backgroud_images.shape[0])
        ]

        ref_result_dict = self.detect_val(
            input_image=adv_com_tensor_rendered_sized.detach(),
            input_path=val_detect_path_list,
            input_file_name='adv_detect_val'
        )

        return epoch_false_pos_rates, ref_result_dict



    def generate_adversarial_com(self,control_image):
        """
        生成对抗伪装
        
            
        """

        """
            ====================================================
            =========== controlnet 的初始化,采样 ===============
            ====================================================
        """

        render_size=self.exp_params['render_face_size']
        # tensor 扩展
        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)
        if control_image.shape[0] != render_size:
            control_image=control_image.expand(render_size,-1,-1,-1) # 扩展到 [render_size, C, H, W]
        # cam_target=[self.exp_params["cam_target_class"]]*self.exp_params["render_face_size"]
        controlnet_adv_texture=self.init_tex_generate(
                                    control_image=control_image,
                                cam_target_class=None,
                                ref_class=None, # object_class
                                negtive_class='car')

        # 保存样本
        
        sample_root=os.path.join(self.exp_params["visual_path"],"controlnet_sample")
        os.makedirs(sample_root,exist_ok=True)
        for i in range(controlnet_adv_texture.shape[0]):
            tensor2picture(controlnet_adv_texture[i],os.path.join(sample_root,f'_{i}.jpg')) 
 


        self.optim_prepare(ini_texture=controlnet_adv_texture)
        self.optim_step()
                        
        return 


    # 验证
    def validate_adversarial_texture(
        self,
        texture_pt_path,
        mesh_path=None,
        save_visual=False
    ):
        """
        标准化验证函数：
        - multi-model evaluation
        - 标准 TP / FP / FN / GT
        - 标准 recall 定义
        """

        if save_visual:
            save_tensor_root = os.path.join(self.exp_params["experiment_path"], "visual_val")
            os.makedirs(save_tensor_root, exist_ok=True)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== 1. 加载 texture ==========
        adv_texture = torch.load(texture_pt_path, map_location=device)

        if isinstance(adv_texture, dict):
            adv_texture = adv_texture["texture"]

        adv_texture = adv_texture.to(device)

        if adv_texture.dim() == 3:
            adv_texture = adv_texture.unsqueeze(0)

        # ========== 2. 加载 3D 模型 ==========
        if mesh_path is None:
            mesh_path = self.exp_params["mesh_model_path"]

        mesh_model, material_list = load_obj_model_return_mesh_material(
            mesh_path,
            device
        )

        target_material = self.exp_params["target_material_dict"]

 

        _ ,val_loader=build_from_cfg(self.exp_params["dataloader"],
                                             batch_size=self.exp_params["batch_size"],
                                              num_workers=self.exp_params["num_workers"],)

        # ========== 4. 每个模型统计 ==========
        model_stats = {}  # model_name -> {tp, fp, fn, gt_total}
        metrices_all = {}  # model_name -> {precision, recall, accuracy, fpr, asr}
        # ========== 5. 遍历数据 ==========
        step_num=0
        for backgroud_images, cameras_pose_path in val_loader:

            backgroud_images = backgroud_images.to(device)

            # ---------- 原始渲染 ----------
            origin_rendered, object_mask = load_parma_and_render_main(
                object_mesh=mesh_model,
                background=backgroud_images,
                path_camera_pose=cameras_pose_path,
                image_size=(
                    self.exp_params["render_size"]["height"],
                    self.exp_params["render_size"]["width"]
                ),
                device=device,
                fov=110,
                blur_radius=0.0,
                faces_per_pixel=1
            )

            # ---------- 纹理映射 ----------
            if target_material is None or len(target_material) != len(adv_texture):
                adv_tex = adv_texture[0]
                target_index_dict = {mat: adv_tex for mat in material_list}
            else:
                target_index_dict = {
                    mat: tex for mat, tex in zip(target_material, adv_texture)
                }

            # ---------- 应用纹理 ----------
            mesh_adv = update_meshes_texture_dict(
                original_meshes_list=mesh_model,
                target_index_dict=target_index_dict,
                material_names_list=material_list,
                device=device
            )

            # ---------- adversarial 渲染 ----------
            adv_rendered, _ = load_parma_and_render_main(
                object_mesh=mesh_adv,
                background=backgroud_images,
                path_camera_pose=cameras_pose_path,
                image_size=(
                    self.exp_params["render_size"]["height"],
                    self.exp_params["render_size"]["width"]
                ),
                device=device,
                fov=110,
                blur_radius=0.0,
                faces_per_pixel=1
            )

            # ---------- resize ----------
            detect_size = self.exp_params["image_size"]


            adv_img = resize_tensor_ratio_pad(adv_rendered, detect_size, detect_size)
            origin_img = resize_tensor_ratio_pad(origin_rendered, detect_size, detect_size)
            mask_resized = resize_tensor_ratio_pad(object_mask, detect_size, detect_size)

            # ---------- GT ----------
            # 根据mask构造gt
            gt_dict = mask_to_gt_dict(
                mask_tensor=mask_resized,
                label=self.exp_params["target_class"],
                num_classes=self.detect_params["nums_class"],
                device=device,
                threshold=1e-3
            )
            #可视化
            if save_visual:
                adv_detect_path_list=[os.path.join(save_tensor_root,f"adv_detect_b{step_num}_{i}") for i in range(adv_img.shape[0])]
                origin_detect_path_list=[os.path.join(save_tensor_root,f"origin_detect_b{step_num}_{i}") for i in range(origin_img.shape[0])]
        
                # 2. 得到列表后，创建每一个路径对应的目录
                for path in adv_detect_path_list + origin_detect_path_list:
                    os.makedirs(path, exist_ok=True)

            else:
                adv_detect_path_list=None
                origin_detect_path_list=None
            step_num+=1
            # 保存tensor为图像

            if save_visual:
                for i in range(adv_img.shape[0]):
                        tensor2picture(adv_rendered[i],os.path.join( adv_detect_path_list[i],"adv.jpg"))
                        tensor2picture(origin_rendered[i],os.path.join( origin_detect_path_list[i],"origin.jpg"))

                    

            # ---------- detection ----------
            result_adv_dict = self.detect_val(
                input_image=adv_img,
                input_path=adv_detect_path_list,
                input_file_name="adv_",
            )

            result_origin_dict = self.detect_val(
                input_image=origin_img,
                input_path=origin_detect_path_list,
                input_file_name="origin_",
            )

            metics_1=camouflage_metrics_lib(adv_img=adv_rendered, 
                                            ori_img=origin_rendered,
                                                bg_img=backgroud_images, 
                                                mask=object_mask)
            for k,v in metics_1.items():
                metrices_all[k]=v+metrices_all.get(k,0)
            # ---------- per-model 统计 ----------
            for model_name in result_origin_dict.keys():

                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "gt_total": 0
                    }

                stats = compute_detection_metrics_v2(
                    pred=result_adv_dict[model_name],
                    gt=gt_dict,
                    iou_thresh=self.exp_params["iou_threshold_val"],
                    conf_thresh=self.exp_params["conf_threshold_val"],
                )


                
                model_stats[model_name]["tp"] += stats["tp"]
                model_stats[model_name]["fp"] += stats["fp"]
                model_stats[model_name]["fn"] += stats["fn"]
                model_stats[model_name]["gt_total"] += stats["gt_total"]


 
        # ========== 6. 汇总 ==========
        metrics_all_models = {}

        print("\n===== Validation Metrics (Per Model) =====")

        for model_name, s in model_stats.items():

            tp = s["tp"]
            fp = s["fp"]
            fn = s["fn"]
            gt_total = s["gt_total"]

            precision = tp / (tp + fp + 1e-6)

            #  标准 recall
            recall = tp / (gt_total + 1e-6)

            accuracy = tp / (tp + fp + fn + 1e-6)

            fpr = fp / (tp + fp + 1e-6)

            #  标准 ASR
            asr =1-recall

            metrics_all_models[model_name] = {
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "false_positive_rate": fpr,
                "attack_success_rate": asr,

                # 附加统计（建议保留）
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_total": gt_total
            }

            print(f"\n--- {model_name} ---")
            for k, v in metrics_all_models[model_name].items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

        # ========== 7. 平均值 ==========
        metrics_avg = {}
        for k ,v in metrices_all.items():
            metrics_avg[k]=v/step_num
        print("\n===== Validation Metrics (Avg) =====")
        for k, v in metrics_avg.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        # ========== 8. 保存结果 ==========
        if self.exp_params["save_result_path"] is not None:
            save_result_path = os.path.join(self.exp_params["save_result_path"], "result.json")
            with open(save_result_path, "w") as f:
                json.dump(metrics_all_models, f, indent=4)
                json.dump(metrics_avg, f, indent=4)
        return metrics_all_models






















