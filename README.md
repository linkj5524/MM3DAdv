# 一、使用说明
## 0. git失败

> * 在windows下，经常会出现git上传仓库，或者git clone 失败的情况。可以按照如下教程进行解决： https://cloud.tencent.com/developer/article/2527142
> * 该教程 大致步骤为：
>     * 1、在windows系统设置代理 
>     * 2、使用git 设置代理。（注意：此代理不会影响代理软件等的正常使用）
## 1. 环境配置
### 1.1 安装依赖


* 方法1：pip install -r requirements.txt
    *  下载与cuda toolkit版本匹配的pytorch，根据自己的cuda toolkit版本下载对应的pytorch安装包，并安装。建议使用国内镜像源：
        pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu124
        
        比如上述命令，后缀cu124  表示版本（如这里表示12.4版本），根据cuda toolkit版本选择，需对应好。
* 方法2（不建议）：安装必要的包，按照environment.yaml 的版本按照，如果显示加载权重的时候，出现key 对不上等问题，则是版本未完全按照要求安装。如果为完全按照版本安装，则也可以按如下修改：

    * 1.1.1 No module named 'pytorch_lightning.utilities.distributed'
        问题原因：pytorch_lightning版本过高，部分函数已经更换过，故会报错
        解决方案：将老版的接口函数，按照新的接口函数修改
        如rom pytorch_lightning.utilities.distributed import rank_zero_only ——> from pytorch_lightning.utilities.rank_zero import rank_zero_only
    * 1.1.2 No module named 'pytorch_lightning.metrics.functional'
### 1.2 环境问题
ldm\modules\diffusionmodules\model.py 下面有个xformers 不能正确安装（windows系统，只有cpu版本的torch才能安装）。如果不想模型加速，则可以不安装，不影响使用。

### 1.3 下载预训练模型
#### 注意事项
* 模型初始化会加载标准的sd模型，以及vae等，上述模型在初始化的时候会直接从huggingface官网下载，国内可能会出现报错。以下是两种解决方法：
    * 1、直接从官网或者其他途径下载模型，并放到huggingface的缓存文件夹下，也可以直接设置缓存文件夹的路径，或者是加载模型时，改成路径。
    * 2、利用国内的镜像，本项目目前使用本方法，更为方便。
        * 2.1  安装依赖：pip install -U huggingface_hub
        * 2.2  设置环境变量：
            * Windows：$env:HF_ENDPOINT = "https://hf-mirror.com"


    
                <span style="color:red">
                在windows 系统下，需要在vscode的终端 运行上述命令，不然不起作用。
                此情况下，如果有代理，需关闭代理
            
            
            
                </span>
        
            * linux： export HF_ENDPOINT=https://hf-mirror.com
#### 相关预训练模型下载
 * 1、VAE模型
 本论文使用的是stabilityai/sd-vae-ft-mse微调 VAE 模型，可直接在 Hugging Face 官网仓库下载：https://huggingface.co/stabilityai/sd-vae-ft-mse。进入页面后点击Files and versions标签页，下载全部必要的配置文件与权重文件，统一存放至某路径，比如./pretrain/sd_vae_ft_mse/

 ```
 XXXX/pretrain/sd_vae_ft_mse/
     config.json
     diffusion_pytorch_model.safetensors (推荐）/diffusion_pytorch_model.bin(备选)
     

 ```

 * 2、ControlNet模型
 本论文使用的是官方预训练的ControlNet模型，可直接在 Hugging Face 官网仓库下载：https://huggingface.co/lllyasviel/ControlNet。进入页面后点击Files and versions标签页，models文件夹中下载canny版本的ControlNet模型下载所需的模型，统一存放至某路径，比如./pretrain/controlnet/
 ```
 XXXX/pretrain/controlnet/control_sd15_canny.pth
 ```    

 * 3、目标检测模型
 yolo系列如果没有下载，会自动下载，找到configs\detect_config\detect_config.yaml文件，找到相关目标检测模型，如果没有下载，则nodel_path 为空即可。其他相关模型到各个模型的官网下载即可。

 * 4、Clip模型
 ControlNet模型需要使用CLIP模型，模型下载地址:  https://huggingface.co/openai/clip-vit-large-patch14.将模型下载到 某目录下，比如./pretrain_model/clip_vit_large_patch14文件夹中即可。
 ```
 XXXX/pretrain_model/clip_vit_large_patch14/
    config.json
    model.safetensors （推荐）/pytorch_model.bin（备选）
    tokenizer.json
    vocab.json
    merges.txt
    tokenizer_config.json
 ```

 * 5、模型注意事项，如果运行时加载模型报错等，可能是模型文件损坏，需要重新下载。

### 1.4 所需数据
#### 1.4.1 3D模型
格式：.obj 或 .obj + .mtl
3D模型也可以自己根据网上开源的blend文件，自己制作，或者从网上开源的blend文件中下载，然后转换为.obj格式。制作时需注意坐标，即模型的朝向等。在./data_tools/3Dmodels文件夹有更详细的处理过程。

#### 1.4.2 训练数据
训练数据可以从直接下载，包含背景图片，以及相机内外参。文件路径
训练数据可以由CARLA仿真平台生成，具体的生成过程见./data_tools/carla_data/
示例如下
```
XXX/train_data/
    train/
       location_000/
          background/
            fixed_000.png
            fixed_001.png
            fixed_002.png
            ....
          camera_pose/
            fixed_000.npz
            fixed_001.npz
            fixed_002.npz
            ....
          rgb/（这个文件不是必要的，调试的时候，比如调试坐标可以参考这个）
            fixed_000.png
            fixed_001.png
            fixed_002.png
            ....
          
        location_001/
          background/
            ....
          camera_pose/
            ....
          rgb/
            ....
        ....
    val/
        location_000/
          background/
            ....
          camera_pose/
            ....
          rgb/
            ....
        ....
```


## 2.训练准备
* configs\adv_config\adv_params.yaml 配置修改
    此文件主要配置训练的参数，如prompt, n_prompt, ddim_steps, guess_mode, strength, scale, scale_optim, seed, eta, save_memory等。如果需要修改，可以按照需求进行修改。涉及ControlNet相关的参数。

* configs\controlnet_config\cldm_v15.yaml 配置修改
    此文件为官方的ControlNet模型的配置文件。如果采用的是预训练的模型，则只需修改model/params/cond_stage_config/params/version 的路径即可。此路径为前面所下载的clip模型路径。比如 XXXX/pretrain_model/clip_vit_large_patch14
* configs\detect_config\detect_config.yaml 配置修改
    这里是训练或验证所涉及的检测模型配置。需要修改不同模型的model_path 和model_type。所使用的模型个数可以自定义增加或删减。
* configs\exp_config\exp_params.yaml 配置修改
    这是实验相关参数，包含训练的超参，损失函数，实验保存的路径等。按照字段名进行修改即可。此外mesh_model_path: 为训练时使用的3D模型路径，target_material_dict: 为训练时使用的材质字典，可以自定义添加或删减。render_face_size 为训练时渲染的面个数。render_face_size 和target_material_dict 需要匹配。同时target_material_dict 为3D模型面的名称，这个可以参考模型的mtl文件，或者在处理3D模型的时候命名好。处理过程见./data_tools/3Dmodels文件夹。
* configs\model_config\models_params.yaml 配置修改
    这是模型路径和模型类型的配置文件。只需修改上面提到的模型，其他为冗余信息，可以不处理。
    主要修改为：
    ```yaml
        # 模型路径配置
        model_paths: 
            controlnet: ControlNet预训练模型路径
            vae_model: VAE预训练模型的路径
        # 模型类型配置
        model_types:
            controlnet: "./configs/controlnet_config/cldm_v15.yaml"   （这个为ControlNet的模型配置文件路径）
    ```

## 3. 训练或推理
### 3.1 训练
* 训练模式
    由于此方法包含使用Canny边缘检测，以及Voronoi边缘检测作为引导信息。这部分可以参考advcom_generate_common_attack.py里面的 ``` if __name__ == '__main__' ```部分。
* 训练命令，
    ```bash
        python advcom_generate_common_attack.py \
        --model_config_path ./configs/model_config/models_params.yaml \
        --exp_config_path ./configs/exp_config/exp_params_cmp.yaml \
        --detect_config_path ./configs/detect_config/detect_config.yaml \
        --adv_config_path ./configs/adv_config/adv_params.yaml
    ```
### 3.2 验证

* 验证命令，
    ```bash
        python advcom_generate_val_common.py \
        --exp_config_path ./configs/exp_config/exp_params_val.yaml \
        --detect_config_path ./configs/detect_config/detect_config.yaml
    ```




