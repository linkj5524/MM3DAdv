"""
完整流程：ShapeNet船只加载 → CARLA环境采样 → Kaolin可微渲染 → 纹理优化 → 1000张验证集生成 → ASR计算
适配GPU：4GB（低显存模式）/6-8GB（推荐）/10GB+（极速）
依赖：carla==0.9.15 kaolin trimesh torch imageio numpy pillow
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import carla
import kaolin as kal
import trimesh
import imageio
from PIL import Image
from tqdm import tqdm

# ====================== 1. 全局配置（只需改这里！） ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 路径配置
SHIP_OBJ_PATH = "./ShapeNet/Ships/02858304/1a2b3c4d/model.obj"  # ShapeNet船只模型路径
INIT_TEXTURE_PATH = "./ocean_texture.jpg"  # 初始纹理路径
VAL_SET_SAVE_PATH = "./val_set"  # 验证集保存路径
# 渲染/优化配置
RENDER_SIZE = 256 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 5e9 else 512
NUM_OPTIM_ITER = 100  # 纹理优化迭代次数
NUM_VAL_IMAGES = 1000  # 验证集图片数量
EPSILON = 0.01  # 纹理扰动上限
LR = 0.001  # 优化器学习率
TARGET_LABEL = 1  # 攻击目标：预测为"非船只"类

# ====================== 2. 低显存优化（自动适配） ======================
def setup_low_memory_mode():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        # 限制显存占用
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        torch.cuda.set_per_process_memory_fraction(0.85)
    print(f"低显存模式已启用 | 渲染分辨率: {RENDER_SIZE}×{RENDER_SIZE}")

# ====================== 3. CARLA环境初始化 ======================
def init_carla():
    """启动CARLA客户端并初始化世界"""
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        # 设置无屏渲染（节省显存）
        settings = world.get_settings()
        settings.no_rendering_mode = True
        world.apply_settings(settings)
        print("CARLA初始化成功！")
        return client, world
    except Exception as e:
        print(f"CARLA启动失败：{e}")
        print("请先运行：./CarlaUE4.sh -RenderOffScreen &")
        sys.exit(1)

# ====================== 4. ShapeNet船只模型加载 ======================
def load_shapenet_ship(ship_obj_path, device):
    """加载并归一化ShapeNet船只模型（适配CARLA单位）"""
    # 加载OBJ模型
    vertices, faces = kal.io.obj.load_mesh(ship_obj_path)
    # 归一化：中心到原点，缩放至CARLA单位（1=1米）
    vertices = vertices - vertices.mean(dim=0)
    vertices = vertices * 0.5  # 缩放至5米以内
    # 转为GPU张量（带batch维度）
    vertices = vertices.float().to(device).unsqueeze(0)
    faces = faces.long().to(device).unsqueeze(0)
    print(f"船只模型加载完成 | 顶点数: {vertices.shape[1]} | 面数: {faces.shape[1]}")
    return vertices, faces

# ====================== 5. CARLA环境参数采样 ======================
def sample_carla_env(world):
    """随机采样CARLA环境参数（天气+相机视角）"""
    # 1. 随机天气
    weather_list = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.FoggyNoon,
        carla.WeatherParameters.RainySunset,
        carla.WeatherParameters.CloudySunset
    ]
    weather = np.random.choice(weather_list)
    world.set_weather(weather)
    
    # 2. 随机相机视角（距离5-10米，方位角0-360°，俯仰角0-30°）
    distance = np.random.uniform(5.0, 10.0)
    azimuth = np.random.uniform(0, 360)
    elevation = np.random.uniform(0, 30)
    
    # 转为相机坐标
    eye_x = distance * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
    eye_y = distance * np.sin(np.radians(elevation))
    eye_z = distance * np.sin(np.radians(azimuth))
    
    eye = torch.tensor([eye_x, eye_y, eye_z], device=DEVICE)
    at = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)  # 船只中心
    up = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    
    return eye, at, up, weather

# ====================== 6. Kaolin渲染器构建 ======================
def build_kaolin_renderer(eye, at, up, render_size):
    """构建Kaolin可微渲染器（对接CARLA视角）"""
    # 相机
    camera = kal.render.Camera.from_args(
        eye=eye, at=at, up=up, fov=60.0,
        width=render_size, height=render_size, device=DEVICE
    )
    # 光照（适配CARLA天气）
    lighting = kal.render.Lighting(
        direction=torch.tensor([1.0, 1.0, 1.0], device=DEVICE),
        color=torch.tensor([1.0, 1.0, 1.0], device=DEVICE),
        ambient_color=torch.tensor([0.5, 0.5, 0.5], device=DEVICE)
    )
    return camera, lighting

# ====================== 7. 纹理加载与优化 ======================
def load_texture(texture_path, device, size):
    """加载初始纹理并开启梯度"""
    img = Image.open(texture_path).convert("RGB").resize((size, size))
    texture = torch.from_numpy(np.array(img)/255.0).float().to(device).unsqueeze(0)
    texture.requires_grad = True
    print(f"初始纹理加载完成 | 尺寸: {size}×{size}")
    return texture

class ShipClassifier(nn.Module):
    """船只分类器（用于计算攻击损失）"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        # 适配渲染分辨率
        fc_in = 64 * (RENDER_SIZE//8) * (RENDER_SIZE//8)
        self.fc = nn.Linear(fc_in, 2)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B,H,W,C) → (B,C,H,W)
        feat = self.conv(x).flatten(1)
        return self.fc(feat)

def optimize_adversarial_texture(vertices, faces, original_texture, world):
    """优化对抗纹理"""
    # 初始化分类器（固定权重，仅用于计算损失）
    classifier = ShipClassifier().to(DEVICE).eval()
    for p in classifier.parameters():
        p.requires_grad = False
    
    # 优化器
    adv_texture = original_texture.clone()
    optimizer = optim.Adam([adv_texture], lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    target = torch.tensor([TARGET_LABEL], device=DEVICE)
    
    # 迭代优化
    print("\n开始优化对抗纹理...")
    loss_history = []
    for iter in tqdm(range(NUM_OPTIM_ITER), desc="纹理优化"):
        # 随机采样环境
        eye, at, up, _ = sample_carla_env(world)
        camera, lighting = build_kaolin_renderer(eye, at, up, RENDER_SIZE)
        
        # 生成UV坐标
        uv_coords = kal.ops.mesh.generate_triangle_uvs(faces, vertices.shape[1])
        uv_coords = uv_coords.to(DEVICE).unsqueeze(0)
        
        # 可微渲染
        render_output = kal.render.mesh.rasterize(
            vertices=vertices, faces=faces, camera=camera,
            uv_coords=uv_coords, textures=adv_texture, lighting=lighting,
            width=RENDER_SIZE, height=RENDER_SIZE, interpolation_mode='bilinear'
        )
        adv_img = torch.clamp(render_output["rgb"], 0.0, 1.0)
        
        # 计算损失
        logits = classifier(adv_img)
        loss = loss_fn(logits, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 约束纹理（数值范围+扰动上限）
        adv_texture.data = torch.clamp(adv_texture.data, 0.0, 1.0)
        adv_texture.data = torch.clamp(
            adv_texture.data,
            original_texture.data - EPSILON,
            original_texture.data + EPSILON
        )
        
        # 记录损失
        loss_history.append(loss.item())
        if (iter+1) % 20 == 0:
            print(f"Iter {iter+1} | Loss: {loss.item():.4f} | Pred: {logits.argmax(1).item()}")
    
    print("纹理优化完成！")
    return adv_texture, loss_history

# ====================== 8. 验证集生成 ======================
def generate_validation_set(vertices, faces, original_texture, adv_texture, world):
    """生成1000张带掩码的验证集图片"""
    # 创建目录
    os.makedirs(f"{VAL_SET_SAVE_PATH}/images", exist_ok=True)
    os.makedirs(f"{VAL_SET_SAVE_PATH}/masks", exist_ok=True)
    
    # 生成掩码函数
    def get_ship_mask(render_output):
        depth = render_output["depth"]
        mask = (depth > 1e-6).float()  # 船只区域=1，背景=0
        return mask.repeat(1, 1, 1, 3)  # (1,H,W,1) → (1,H,W,3)
    
    # 张量转图像
    def tensor2img(tensor):
        return (tensor[0].cpu().detach().numpy() * 255).astype(np.uint8)
    
    # 批量生成
    print("\n开始生成1000张验证集...")
    for idx in tqdm(range(NUM_VAL_IMAGES), desc="验证集生成"):
        # 随机环境+渲染器
        eye, at, up, _ = sample_carla_env(world)
        camera, lighting = build_kaolin_renderer(eye, at, up, RENDER_SIZE)
        uv_coords = kal.ops.mesh.generate_triangle_uvs(faces, vertices.shape[1])
        uv_coords = uv_coords.to(DEVICE).unsqueeze(0)
        
        # 渲染原始纹理
        render_ori = kal.render.mesh.rasterize(
            vertices=vertices, faces=faces, camera=camera,
            uv_coords=uv_coords, textures=original_texture, lighting=lighting,
            width=RENDER_SIZE, height=RENDER_SIZE
        )
        # 渲染对抗纹理
        render_adv = kal.render.mesh.rasterize(
            vertices=vertices, faces=faces, camera=camera,
            uv_coords=uv_coords, textures=adv_texture, lighting=lighting,
            width=RENDER_SIZE, height=RENDER_SIZE
        )
        # 生成掩码
        mask = get_ship_mask(render_ori)
        
        # 保存图片
        imageio.imsave(f"{VAL_SET_SAVE_PATH}/images/{idx}_ori.png", tensor2img(render_ori["rgb"]))
        imageio.imsave(f"{VAL_SET_SAVE_PATH}/images/{idx}_adv.png", tensor2img(render_adv["rgb"]))
        imageio.imsave(f"{VAL_SET_SAVE_PATH}/masks/{idx}_mask.png", tensor2img(mask))
    
    print(f"验证集生成完成！路径: {VAL_SET_SAVE_PATH}")

# ====================== 9. 攻击准确率（ASR）计算 ======================
def calculate_attack_success_rate(val_set_path):
    """计算验证集攻击准确率"""
    classifier = ShipClassifier().to(DEVICE).eval()
    asr = 0
    print("\n开始计算攻击准确率...")
    for idx in tqdm(range(NUM_VAL_IMAGES), desc="ASR计算"):
        # 加载对抗纹理图片
        img_path = f"{val_set_path}/images/{idx}_adv.png"
        img = imageio.imread(img_path)
        img = torch.from_numpy(img/255.0).float().to(DEVICE).unsqueeze(0)
        
        # 预测
        logits = classifier(img)
        if logits.argmax(1).item() == TARGET_LABEL:
            asr += 1
    
    asr_rate = asr / NUM_VAL_IMAGES
    print(f"\n攻击准确率（ASR）: {asr_rate * 100:.2f}%")
    return asr_rate

# ====================== 10. 主函数（一键运行） ======================
def main():
    # 1. 初始化配置
    setup_low_memory_mode()
    
    # 2. 启动CARLA
    client, world = init_carla()
    
    # 3. 加载船只模型
    vertices, faces = load_shapenet_ship(SHIP_OBJ_PATH, DEVICE)
    
    # 4. 加载初始纹理
    original_texture = load_texture(INIT_TEXTURE_PATH, DEVICE, RENDER_SIZE)
    
    # 5. 优化对抗纹理
    adv_texture, loss_history = optimize_adversarial_texture(vertices, faces, original_texture, world)
    
    # 6. 生成验证集
    generate_validation_set(vertices, faces, original_texture, adv_texture, world)
    
    # 7. 计算攻击准确率
    asr = calculate_attack_success_rate(VAL_SET_SAVE_PATH)
    
    # 8. 保存优化后的纹理
    adv_texture_np = adv_texture[0].cpu().detach().numpy()
    Image.fromarray((adv_texture_np * 255).astype(np.uint8)).save("./adv_texture.png")
    print("\n所有流程完成！")
    print(f"→ 优化后的纹理已保存：./adv_texture.png")
    print(f"→ 验证集路径：{VAL_SET_SAVE_PATH}")
    print(f"→ 攻击准确率：{asr*100:.2f}%")

if __name__ == "__main__":
    main()