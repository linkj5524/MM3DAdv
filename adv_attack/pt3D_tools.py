import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform,
    TexturesVertex,
    PerspectiveCameras
)
import numpy as np
from PIL import Image
import os
import cv2

def visualize_and_save_render(
    image_tensor: torch.Tensor,  # 输入B×3×H×W的张量
    save_dir: str = "debug_results",
    title_prefix: str = "OBJ Model Render (Real Camera Params)"
):
    """
    可视化并保存批量渲染的图像
    参数：
        image_tensor: 渲染输出的张量，形状为(B, 3, H, W)，数值范围[0,1]（float32）
        save_dir: 保存路径
        title_prefix: 图像标题前缀
    """
    # 1. 创建保存目录（不存在则创建）
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 张量转numpy并处理维度/数据类型
    # - 移到CPU → 转numpy → 调整维度(B,3,H,W)→(B,H,W,3) → 确保数值范围[0,1]
    image_np = image_tensor.cpu().clamp(0.0, 1.0).numpy()  # 限制范围，避免异常值
    image_np = np.transpose(image_np, (0, 2, 3, 1))  # 维度转换：B×3×H×W → B×H×W×3
    
    # 3. 循环可视化并保存每张图像
    for i in range(len(image_np)):
        # 单张图像（H,W,3）
        single_img = image_np[i]
        
        # 可视化
        plt.figure(figsize=(8, 6))
        plt.imshow(single_img)
        plt.title(f"{title_prefix} {i}")
        plt.axis("off")
        plt.tight_layout()  # 去除边距，避免标题被裁剪
        plt.show()
        
        # 保存图像（自动将[0,1]的float转为[0,255]的uint8）
        save_path = os.path.join(save_dir, f"render_obj_{i}.png")
        plt.imsave(save_path, single_img)
        print(f"渲染结果已保存至：{save_path}")



def load_background_images(
    bg_paths: list,
    target_size: tuple = None,  # (width, height)，可选：统一图像尺寸
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    加载背景图片列表，返回标准化的张量
    参数：
        bg_paths: 背景图片路径列表（如 ["./rgb_000.png", "./rgb_001.png"]）
        target_size: 可选，统一图像尺寸 (width, height)，None则保留原图尺寸
        device: 张量存储设备（cpu/cuda）
    返回：
        torch.Tensor: 形状为 (B, C, H, W) 的张量，数值范围 [0,1]，float32类型
                      C=3（RGB），B=len(bg_paths)
    """
    bg_tensors = []
    
    for idx, bg_path in enumerate(bg_paths):
        # 1. 校验文件是否存在
        if not os.path.exists(bg_path):
            raise FileNotFoundError(f"背景图片不存在：{bg_path}")
        
        # 2. 加载图片（RGB模式）
        try:
            img = Image.open(bg_path).convert("RGB")  # 强制转为RGB，避免RGBA/灰度图
        except Exception as e:
            raise RuntimeError(f"加载图片失败 {bg_path}：{str(e)}")
        
        # 3. 统一尺寸（可选）
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)  # 高质量缩放
        
        # 4. 转换为numpy数组 → (H, W, 3)，数值范围 [0,255] uint8
        img_np = np.array(img, dtype=np.uint8)
        
        # 5. 转换为张量并标准化
        # - (H,W,3) → (3,H,W)
        # - uint8 [0,255] → float32 [0,1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        
        # 6. 添加到列表
        bg_tensors.append(img_tensor)
    
    # 7. 拼接为批量张量 (B, C, H, W)
    batch_tensor = torch.stack(bg_tensors, dim=0).to(device)
    
    return batch_tensor





############################################################
############################################################
############################################################


def paste_non_white_regions(
    rendered_images: torch.Tensor,  # 渲染图像张量 (B, 3, H, W)，范围[0,1]
    background_images: torch.Tensor,  # 背景图像张量 (B, 3, H, W)，范围[0,1]
    white_threshold: float = 0.99,  # 判定阈值：三通道均>该值则为白色
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    将渲染图像的非白色区域（RGB三通道不同时为1）粘贴到背景图像上
    核心逻辑：
        1. 判定条件：渲染图像中 RGB三个通道值均 > white_threshold → 白色区域
        2. 非白色区域用渲染图像像素，白色区域保留背景图像像素
    参数：
        rendered_images: 渲染图像张量，形状(B, 3, H, W)，float32，范围[0,1]
        background_images: 背景图像张量，形状(B, 3, H, W)，float32，范围[0,1]
        white_threshold: 白色判定阈值（0-1），越接近1越严格（0.99≈纯黑/纯白渲染场景）
        device: 计算设备（cpu/cuda）
    返回：
        torch.Tensor: 融合后的图像张量，形状(B, 3, H, W)，范围[0,1]，float32
    """

    # ================= 2 严格判定白色区域（三通道均>阈值） =================
    # 步骤1：判断每个通道是否大于阈值 → (B, 3, H, W) 的布尔张量
    is_channel_white = rendered_images > white_threshold
    # 步骤2：三通道同时满足（均>阈值）→ 白色区域，结果为 (B, 1, H, W)
    is_white_region = torch.all(is_channel_white, dim=1, keepdim=True)
    # 步骤3：转换为float掩码 → 白色区域=1，非白色区域=0
    white_mask = is_white_region.float()
    # 非白色区域掩码 → 非白色=1，白色=0（用于粘贴渲染图）
    non_white_mask = 1.0 - white_mask

    # ================= 3 图像融合（粘贴非白色区域） =================
    # 非白色区域：渲染图像素 × 非白色掩码
    # 白色区域：背景图像素 × 白色掩码
    fused_images = (
        rendered_images * non_white_mask + 
        background_images * white_mask
    )
    
    # ================= 4 限制数值范围（避免浮点溢出） =================
    fused_images = torch.clamp(fused_images, 0.0, 1.0)
    
    return fused_images.to(device)




############################################################
############################################################
############################################################

# ================= 加载图像类数据 =================
def load_rgb(path):
    """
    加载RGB图片（使用cv2读取）
    :param path: 图片完整路径（如 "./data/rgb/001.png"）
    :return: np.array (H, W, 3) RGB格式的图像数组，失败返回None
    """

    # 1. 使用cv2读取图片（默认BGR格式）
    img_bgr = cv2.imread(path)
    if img_bgr is None:  # 覆盖文件不存在/格式错误/路径错误等场景
        print(f"错误：无法读取RGB文件 {path}（文件不存在或格式错误）")
        return None
    
    # 2. 转换为标准RGB格式（cv2默认BGR，需转换以符合常规RGB认知）
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb


def load_depth(path):
    """
    加载深度图（还原为深度值矩阵，单位：米）
    :param path: 深度图完整路径（如 "./data/depth/001.png"）
    :return: np.array (H, W) 深度值矩阵
    """

    # 读取深度图（Carla保存的深度图为8位RGB编码，需解码）
    img = Image.open(path).convert("RGB")
    depth_array = np.array(img, dtype=np.float32)
    
    # Carla深度图解码公式（官方标准）
    # R/G/B分别对应深度值的低/中/高位，组合为0~1的归一化值，再还原为实际深度
    normalized_depth = depth_array[:, :, 0] + depth_array[:, :, 1] * 256 + depth_array[:, :, 2] * 256 * 256
    normalized_depth = normalized_depth / (256 * 256 * 256 - 1)
    actual_depth = 1000 * normalized_depth  # Carla默认深度范围0~1000米
    
    return actual_depth



def load_mask(path):
    """

    提取最大连通区域为1，其余为0
    :param path: 掩码图完整路径（如 "./data/mask/001.png"）
    :return: np.array (H, W) 二值矩阵（1=最大目标连通区，0=其余区域），失败返回None
    """

    # 1. 使用cv2读取图片（BGR格式），并转换为RGB格式
    img = cv2.imread(path)
    if img is None:  # 处理cv2读取失败
        print(f"错误：无法读取掩码文件 {path}（文件不存在或格式错误）")
        return None
    
    # cv2读取的是BGR，需转为RGB以便按R/G/B通道判断
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # 筛选 B=255, R=0, G=0 的纯蓝色像素（若实际要蓝色，取消下面注释并注释上面红色规则）
    blue_pixels = (img_rgb[:, :, 2] == 142) & (img_rgb[:, :, 0] == 0) & (img_rgb[:, :, 1] == 0)
    target_mask = blue_pixels.astype(np.uint8) * 255

    # 3. 查找所有目标连通区域（轮廓）
    # RETR_EXTERNAL：只找最外层轮廓，CHAIN_APPROX_SIMPLE：压缩轮廓点
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print(f"警告：文件 {path} 中未检测到目标颜色区域（R=255,G=0,B=0），返回全0矩阵")
        return np.zeros_like(target_mask)

    # 4. 找到面积最大的目标连通区域
    max_contour = max(contours, key=cv2.contourArea)

    # 5. 创建二值矩阵：最大目标区域为1，其余为0
    result = np.zeros_like(target_mask, dtype=np.uint8)  # 初始全0
    cv2.drawContours(result, [max_contour], -1, 1, thickness=cv2.FILLED)  # 填充最大区域为1

    return result



# ================= 加载几何类数据 =================
def load_camera_pose(path):
    """
    加载相机位姿
    :param path: 位姿文件完整路径（如 "./data/camera_pose/001.npz"）
    :return: dict {
        "location": [x, y, z],  # 位置
        "rotation": [pitch, yaw, roll]  # 旋转（单位：度）
    }
    """

    # 加载npz文件
    data = np.load(path)
    pose = data["pose"].astype(np.float32)
    
    # 还原为位置和旋转
    camera_pose = {
        "location": [pose[0], pose[1], pose[2]],  # x, y, z
        "rotation": [pose[3], pose[4], pose[5]]   # pitch, yaw, roll
    }
    data.close()  # 关闭文件句柄
    return camera_pose


def load_camera_intrinsics(path):
    """
    加载相机内参
    :param path: 内参文件完整路径（如 "./data/camera_intrinsics/001.npz"）
    :return: np.array (3, 3) 相机内参矩阵K
    """

    # 加载npz文件
    data = np.load(path)
    K = data["K"].astype(np.float32)
    data.close()  # 关闭文件句柄
    return K

    


def load_obj_model(obj_path: str, device: torch.device) :
    """
    从指定路径加载OBJ模型，返回PyTorch3D的Meshes对象（与原有sphere模型格式一致）
    
    参数:
        obj_path: OBJ文件的绝对/相对路径
        device: 模型加载的设备 (cpu/cuda)
    
    返回:
        Meshes对象: 包含顶点、面、默认纹理的3D模型
    """
    # 1. 加载OBJ文件（自动处理mtl/纹理，若无纹理则后续初始化）
    try:
        # load_objs_as_meshes会自动解析OBJ+MTL，返回Meshes对象
        mesh = load_objs_as_meshes(
            [obj_path],          # 传入路径列表（单文件）
            device=device,
            load_textures=True,  # 尝试加载纹理（无纹理时返回None）
            create_texture_atlas=True  # 兼容不同纹理格式
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"OBJ文件未找到，请检查路径: {obj_path}")
    except Exception as e:
        raise RuntimeError(f"加载OBJ失败: {str(e)}")

    # 2. 确保模型有默认顶点颜色（兼容原有代码逻辑）
    if mesh.textures is None:
        # 若无纹理，初始化和原有sphere一致的蓝色系顶点颜色
        verts = mesh.verts_packed()
        default_color = torch.ones_like(verts)[None] * torch.tensor([0.2, 0.7, 1.0], device=device)
        mesh.textures = TexturesVertex(verts_features=default_color)

    # 3. 返回标准化的Meshes对象
    return mesh



# def generate_camera_from_params(
#     pose_path: str, 
#     intrinsics_path: str, 
#     device: torch.device,
#     img_size: tuple = (640, 480)  # (width, height)
# ) -> PerspectiveCameras:
#     """
#     根据相机内参、外参文件路径生成PyTorch3D的PerspectiveCameras对象
#     参数:
#         pose_path: 相机外参（位姿）文件路径（npz格式）
#         intrinsics_path: 相机内参文件路径（npz格式）
#         device: 运行设备 (cpu/cuda)
#         img_size: 图像尺寸 (width, height)，用于计算fov
#     返回:
#         PerspectiveCameras: 配置好的相机对象
#     """
#     # 1. 加载内外参
#     pose = load_camera_pose(pose_path)
#     K_np = load_camera_intrinsics(intrinsics_path)
    
#     # 2. 解析外参：转换旋转（欧拉角→旋转矩阵）、提取平移向量
#     # 欧拉角（pitch,yaw,roll，单位度）转旋转矩阵（PyTorch3D默认右手系）
#     pitch, yaw, roll = np.radians(pose["rotation"])
    
#     # 计算旋转矩阵（Z-Y-X顺序，适配Carla/常规相机坐标系）
#     cos_p, sin_p = np.cos(pitch), np.sin(pitch)
#     cos_y, sin_y = np.cos(yaw), np.sin(yaw)
#     cos_r, sin_r = np.cos(roll), np.sin(roll)
    
#     # 旋转矩阵 R (3x3)
#     R_x = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])
#     R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
#     R_z = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
#     R = R_z @ R_y @ R_x  # 组合旋转矩阵
#     R = np.transpose(R)  # 适配PyTorch3D的坐标系方向
#     R = torch.tensor(R, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)
    
#     # 平移向量 T (相机位置，PyTorch3D中T是相机在世界坐标系的位置，符号需调整)
#     T = torch.tensor(pose["location"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3)
    
#     # 3. 解析内参：适配PerspectiveCameras的参数格式
#     fx = K_np[0, 0]
#     fy = K_np[1, 1]
#     cx = K_np[0, 2]
#     cy = K_np[1, 2]
#     width, height = img_size
    
#     # 4. 构造PerspectiveCameras所需的参数
#     # focal_length: (N, 2) 格式，对应fx, fy
#     focal_length = torch.tensor([[fx, fy]], dtype=torch.float32, device=device)
    
#     # principal_point: (N, 2) 格式，对应cx, cy（主点坐标）
#     principal_point = torch.tensor([[cx, cy]], dtype=torch.float32, device=device)
    
#     # 可选：构造完整的K矩阵（4x4），如果提供则无需指定focal_length和principal_point
#     K = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
#     K[0, :3, :3] = torch.tensor(K_np, dtype=torch.float32, device=device)
    
#     # 图像尺寸：(N, 2) 格式，(height, width)
#     image_size = torch.tensor([[height, width]], dtype=torch.int32, device=device)
    
#     # 5. 创建PerspectiveCameras对象（两种方式选其一）
#     # 方式1：使用focal_length和principal_point（推荐）
#     cameras = PerspectiveCameras(
#         focal_length=focal_length,
#         principal_point=principal_point,
#         R=R,
#         T=T,
#         device=device,
#         in_ndc=False,  # 屏幕空间坐标，设置为False
#         image_size=image_size
#     )
    
    
#     return cameras

def generate_camera_from_params(
    pose_paths: list,  
    device: torch.device,
    fov: float = 110.0,  
    img_size: tuple = (640, 480)  # (width, height)
) -> FoVPerspectiveCameras:  # 修正返回类型：PerspectiveCameras → FoVPerspectiveCameras
    """
    批量加载位姿文件，生成单个批量相机对象（而非列表）
    参数：
        pose_paths: 相机位姿文件路径列表
        device: 运行设备 (cpu/cuda)
        fov: 相机视场角（单位：度），默认110°
        img_size: 图像尺寸 (width, height)，默认(640,480)
    返回：
        FoVPerspectiveCameras: 批量相机对象（包含len(pose_paths)个相机）
    """
    # ================= 1 初始化存储列表 =================
    R_list = []  # 存储所有相机的旋转矩阵 (3,3)
    T_list = []  # 存储所有相机的平移向量 (3,)

    # ================= 2 批量加载位姿并计算外参 =================
    for pose_path in pose_paths:
        # 加载单个位姿文件
        pose = load_camera_pose(pose_path)
        if pose is None:
            raise ValueError(f"位姿文件加载失败：{pose_path}")
        
        # 提取相机位置
        pt3D_x, pt3D_z, pt3D_y = pose['location']
        
        # 计算俯仰角和方位角（修正坐标系转换）
        pitch = np.degrees(np.arctan2(pt3D_y, np.sqrt(pt3D_x**2 + pt3D_z**2)))
        # 3D torch ,pytoch3D 和carla 坐标系关系，需如下调整
        azim_angle = np.degrees(np.arctan2(pt3D_x, pt3D_z))  
        azim_angle = -azim_angle  # 符号修正
        
        # 生成单个相机的外参
        R_single, T_single = look_at_view_transform(
            dist=4.0,
            elev=pitch,
            azim=azim_angle
        )
        
        # 转换为tensor并移到指定设备，去除batch维度
        R_single = R_single.squeeze(0).to(device)  # (3,3)
        T_single = T_single.squeeze(0).to(device)  # (3,)
        
        # 添加到列表
        R_list.append(R_single)
        T_list.append(T_single)

    # ================= 3 拼接为批量外参 =================
    # 拼接为 (B, 3, 3) 旋转矩阵（B=相机数量）
    R_batch = torch.stack(R_list, dim=0)
    # 拼接为 (B, 3) 平移向量
    T_batch = torch.stack(T_list, dim=0)

    # ================= 4 生成批量相机对象 =================
    batch_cameras = FoVPerspectiveCameras(
        device=device,
        R=R_batch,                  # 批量旋转矩阵 (B, 3, 3)
        T=T_batch,                  # 批量平移向量 (B, 3)
        fov=fov,                    # 所有相机共用的视场角
        # aspect_ratio=img_size[0]/img_size[1],  # 宽高比
        # 显式设置图像尺寸，确保和渲染配置匹配
        
    )

    return batch_cameras
def camera_generate_fixed(device):
        # --------------------------------
    # 2 设置相机
    # --------------------------------
    elev = np.degrees(np.arctan2(0.8, 4))  # 仰角（基于CAM_HEIGHT和CAM_RADIUS计算）
    R, T = look_at_view_transform(
        dist=4,
        elev=elev,
        azim=0
    )

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=110,
        
    )
    return cameras

def model_generate_fixed(device):

    # --------------------------------
    # 1 生成自带模型 (sphere)
    # --------------------------------
    mesh = ico_sphere(level=3, device=device)

    # 给顶点设置颜色
    verts = mesh.verts_packed()
    color = torch.ones_like(verts)[None] * torch.tensor([0.2, 0.7, 1.0], device=device)

    mesh.textures = TexturesVertex(verts_features=color)
    return mesh


def light_set_fixed(device):
        # --------------------------------
    # 3 光照
    # --------------------------------
    lights = PointLights(
        device=device,
        location=[[2.0, 2.0, -2.0]]
    )
    return lights


def rasterizer_set():
    # --------------------------------
    # 4 Rasterizer
    # --------------------------------
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    return raster_settings



def render_process(
    cameras,  # 批量相机对象（FoVPerspectiveCameras），而非list
    raster_settings,
    lights,
    mesh,
    device: torch.device
) -> torch.Tensor:
    """
    为批量相机渲染图像，返回B×C×H×W的张量
    参数：
        cameras: PyTorch3D的批量相机对象（FoVPerspectiveCameras/PerspectiveCameras）
        raster_settings: RasterizationSettings对象（光栅化配置）
        lights: 光源对象（如PointLights）
        mesh: 待渲染的Meshes对象（3D模型）
        device: 运行设备 (cpu/cuda)
    返回：
        torch.Tensor: 形状为 (B, 3, H, W) 的张量（B=相机数，C=3，H/W=图像尺寸）
                      数值范围 [0, 1]，float32类型
    """
    # ================= 1 初始化批量渲染器 =================
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,          # 批量相机对象（自动适配多相机）
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,          # 批量相机
            lights=lights             # 光源（若需批量光源，可传入形状为(B,3)的lights）
        )
    )

    # ================= 2 批量渲染 =================
    # 渲染结果：形状为 (B, H, W, 4) → 4通道（RGB+Alpha）
    # mesh会自动广播到和cameras相同的batch维度
    images = renderer(mesh)

    # ================= 3 格式转换 =================
    # 1. 提取RGB通道（丢弃Alpha）：(B, H, W, 3)
    rgb_images = images[..., :3]
    # 2. 调整维度顺序：(B, H, W, 3) → (B, 3, H, W)（符合PyTorch张量规范）
    rgb_images = rgb_images.permute(0, 3, 1, 2)
    # 3. 确保数值范围在[0,1]（避免渲染溢出）
    rgb_images = torch.clamp(rgb_images, 0.0, 1.0)

    return rgb_images



def load_parma_and_render_main(object_mesh,
                               backgroud,
                               path_camera_pose,
                               image_size,
                               device,
                               fov=110,
                               blur_radius=0.0,
                               faces_per_pixel=1,
                               white_threshold=0.999):
    '''
    object_mesh: 3D模型
    backgroud: 背景图,多batch,数据范围0-1
    path_camera_pose: 相机位姿文件路径 的列表
    image_size: 图像尺寸
    device: 运行设备
    fov: 视场角
    blur_radius: 光栅化模糊半径
    faces_per_pixel: 光栅化每像素面数
    '''

    cameras = generate_camera_from_params(
        pose_paths=path_camera_pose,
        device=device,
        fov=fov,
        img_size=image_size
    )


    # -------------------------- 4. 初始化渲染组件 --------------------------
    # 固定光照（复用原有函数）
    light = light_set_fixed(device)

    # 光栅化设置（适配图像尺寸）
    raster_settings = RasterizationSettings(
        image_size=image_size,  # 高度
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel
    )


    image_tensor = render_process(
        cameras=cameras,
        raster_settings=raster_settings,
        lights=light,
        mesh=object_mesh,
        device=device
    )

    # 后处理
    rendered_image_tensor = paste_non_white_regions(image_tensor, backgroud,
                                            white_threshold=white_threshold)

    # --------------------------  结果可视化与保存(debug) --------------------------
    visualize_and_save_render(image_tensor)
    visualize_and_save_render(rendered_image_tensor,save_dir="debug_results/1")

    return image_tensor


def main_debug2():

    # -------------------------- 1. 配置路径与参数 --------------------------
    # 数据路径（按你的需求指定）
    root="/root/autodl-fs/data/data_test/carla_data/vehicle_tesla_model3/location_000/"
    name="fixed_000"
    RGB_PATH = os.path.join(root, "rgb", name+".png")       # RGB图路径
    DEPTH_PATH = os.path.join(root, "depth", name+".png")   # 深度图路径
    MASK_PATH = os.path.join(root, "mask", name+".png")     # 掩码图路径
    POSE_PATH = os.path.join(root, "camera_pose", name+".npz")   # 位姿文件路径
    INTRINSICS_PATH = os.path.join(root, "camera_intrinsics", name+".npz")  # 内参文件路径
    SAVE_DIR = "./debug_results/exp2"  # 结果保存目录
    OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/TeslaModel3_blue.obj'  # OBJ模型路径

    # 图像尺寸（从RGB图自动获取，也可手动指定）
    IMG_SIZE = ( 720,1280)  # ( height,width)，若需自动获取可参考下方注释代码

    # -------------------------- 2. 初始化设备与路径检查 --------------------------
    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存目录
    

    # -------------------------- 3. 加载模型与相机参数 --------------------------


    mesh_model = load_obj_model(OBJECT_OBJ_PATH, device)

    camera_paths_list=[POSE_PATH,POSE_PATH.replace("fixed_000","fixed_001")] 
    backgroud_paths_list=[RGB_PATH,RGB_PATH.replace("fixed_000","fixed_001")]

    backgroud_images=load_background_images(backgroud_paths_list,device=device)

    new_mesh = mesh_model.extend(len(camera_paths_list))
    
    images_rnedered = load_parma_and_render_main(object_mesh=new_mesh,
                                                 backgroud=backgroud_images,
                                                 path_camera_pose=camera_paths_list,
                                                 image_size=IMG_SIZE,
                                                 device=device,
                                                 fov=110,
                                                 blur_radius=0.0,
                                                 faces_per_pixel=1)
    
    #
    

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mesh_model=model_generate_fixed(device)
    camera=camera_generate_fixed(device)
    light=light_set_fixed(device)
    Rasterizer=rasterizer_set()
    image=render_process( cameras=camera, 
                         raster_settings= Rasterizer,
                         lights= light,
                         mesh=mesh_model,
                         device=device)

    # --------------------------------
    # 7 显示
    # --------------------------------
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    plt.imsave("render_sphere.png", image)


def main_debug1():

    # -------------------------- 1. 配置路径与参数 --------------------------
    # 数据路径（按你的需求指定）
    root="/root/autodl-fs/data/data_test/carla_data/vehicle_tesla_model3/location_000/"
    name="fixed_002"
    RGB_PATH = os.path.join(root, "rgb", name+".png")       # RGB图路径
    DEPTH_PATH = os.path.join(root, "depth", name+".png")   # 深度图路径
    MASK_PATH = os.path.join(root, "mask", name+".png")     # 掩码图路径
    POSE_PATH = os.path.join(root, "camera_pose", name+".npz")   # 位姿文件路径
    INTRINSICS_PATH = os.path.join(root, "camera_intrinsics", name+".npz")  # 内参文件路径
    SAVE_DIR = "./debug_results/exp2"  # 结果保存目录
    OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/TeslaModel3_blue.obj'  # OBJ模型路径

    # 图像尺寸（从RGB图自动获取，也可手动指定）
    IMG_SIZE = ( 720,1280)  # ( height,width)，若需自动获取可参考下方注释代码

    # -------------------------- 2. 初始化设备与路径检查 --------------------------
    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存目录
    

    # -------------------------- 3. 加载模型与相机参数 --------------------------
    # 加载OBJ模型（替换原有球体模型）
    print(f"加载OBJ模型：{OBJECT_OBJ_PATH}")
    try:
        mesh_model = load_obj_model(OBJECT_OBJ_PATH, device)
    except Exception as e:
        print(f"加载OBJ模型失败：{e}")
        return

    # 加载相机参数并生成相机（替换原有固定相机）
    print(f"加载相机参数：\n  外参：{POSE_PATH}\n  内参：{INTRINSICS_PATH}")
    try:
        camera = generate_camera_from_params(
            pose_path=POSE_PATH,
            intrinsics_path=INTRINSICS_PATH,
            device=device,
            img_size=IMG_SIZE
        )
    except Exception as e:
        print(f"生成相机失败：{e}")
        return

    # -------------------------- 4. 初始化渲染组件 --------------------------
    # 固定光照（复用原有函数）
    light = light_set_fixed(device)

    # 光栅化设置（适配图像尺寸）
    raster_settings = RasterizationSettings(
        image_size=IMG_SIZE,  # 高度
        blur_radius=0.0,
        faces_per_pixel=1
    )

    # -------------------------- 5. 执行渲染 --------------------------
    print("开始渲染模型...")
    try:
        image = render_process(
            cameras=camera,
            raster_settings=raster_settings,
            lights=light,
            mesh=mesh_model,
            device=device
        )
    except Exception as e:
        print(f"渲染失败：{e}")
        return

    # -------------------------- 6. 结果可视化与保存 --------------------------
    # 显示渲染结果
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("OBJ Model Render (Real Camera Params)")
    plt.axis("off")
    plt.show()

    # 保存渲染结果
    save_path = os.path.join(SAVE_DIR, "render_tesla.png")
    plt.imsave(save_path, image)
    print(f"渲染结果已保存至：{save_path}")

    # （可选）加载并显示原始RGB图对比
    if os.path.exists(RGB_PATH):
        rgb_img = load_rgb(RGB_PATH)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title("Original RGB Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title("Rendered OBJ Model")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # 保存对比图
        compare_save_path = os.path.join(SAVE_DIR, "rgb_vs_render.png")
        plt.savefig(compare_save_path, bbox_inches='tight', dpi=150)
        print(f"RGB与渲染结果对比图已保存至：{compare_save_path}")

if __name__ == "__main__":
    main_debug2()
    # # ---------------- 配置区（修改为你的文件路径） ----------------
    # RGB_PATH = "/root/autodl-fs/data/data_test/location_000_random/rgb/random_000.png"       # RGB图路径
    # DEPTH_PATH = "/root/autodl-fs/data/data_test/location_000_random/depth/random_000.png"   # 深度图路径
    # MASK_PATH = "/root/autodl-fs/data/data_test/location_000_random/mask/random_000.png"     # 掩码图路径
    # POSE_PATH = "/root/autodl-fs/data/data_test/location_000_random/camera_pose/random_000.npz"   # 位姿文件路径
    # INTRINSICS_PATH = "/root/autodl-fs/data/data_test/location_000_random/camera_intrinsics/random_000.npz"  # 内参文件路径
    # SAVE_DIR = "./debug_results"  # 结果保存目录
    # OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/TeslaModel3.obj'  # OBJ模型路径
    
    # # 创建保存目录
    # os.makedirs(SAVE_DIR, exist_ok=True)
    
    # # 设置设备（优先GPU）
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")

    # # ---------------- 加载数据并打印信息 ----------------
    # print("="*50)
    # print("开始加载数据...")
    # print("="*50)

    # # 1. 加载RGB
    # rgb = load_rgb(RGB_PATH)
    # print(f"\n【RGB信息】")
    # if rgb is not None:
    #     print(f"形状: {rgb.shape} | 数据类型: {rgb.dtype} | 像素值范围: {rgb.min()}~{rgb.max()}")
    #     # 保存RGB图
    #     rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(os.path.join(SAVE_DIR, "rgb.png"), rgb_bgr)
    #     print(f"✅ RGB图已保存到 {os.path.join(SAVE_DIR, 'rgb.png')}")
    # else:
    #     print("加载失败")

    # # 2. 加载深度
    # depth = load_depth(DEPTH_PATH)
    # print(f"\n【深度图信息】")
    # if depth is not None:
    #     print(f"形状: {depth.shape} | 数据类型: {depth.dtype}")
    #     print(f"深度范围: {depth.min():.2f} ~ {depth.max():.2f} 米")
    #     # 保存深度可视化图+原始数据
    #     depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
    #     depth_norm = depth_norm.astype(np.uint8)
    #     cv2.imwrite(os.path.join(SAVE_DIR, "depth_vis.png"), depth_norm)
    #     np.save(os.path.join(SAVE_DIR, "depth_raw.npy"), depth)
    #     print(f"✅ 深度图（可视化+原始数据）已保存到 {SAVE_DIR}")
    # else:
    #     print("加载失败")

    # # 3. 加载掩码
    # mask = load_mask(MASK_PATH)
    # print(f"\n【掩码图信息】")
    # if mask is not None:
    #     print(f"形状: {mask.shape} | 数据类型: {mask.dtype}")
    #     print(f"掩码像素数: {np.sum(mask)} | 取值范围: {mask.min()}~{mask.max()}")
    #     # 保存掩码图
    #     mask_vis = mask * 255
    #     cv2.imwrite(os.path.join(SAVE_DIR, "mask.png"), mask_vis)
    #     print(f"✅ 掩码图已保存到 {os.path.join(SAVE_DIR, 'mask.png')}")
    # else:
    #     print("加载失败")

    # # 4. 加载相机位姿
    # pose = load_camera_pose(POSE_PATH)
    # print(f"\n【相机位姿信息】")
    # if pose is not None:
    #     print(f"位置 (x,y,z): {[round(v, 3) for v in pose['location']] if pose['location'] is not None else '无'}")
    #     print(f"旋转 (pitch,yaw,roll): {[round(v, 3) for v in pose['rotation']] if pose['rotation'] is not None else '无'}")
    # else:
    #     print("加载失败")

    # # 5. 加载相机内参
    # intrinsics = load_camera_intrinsics(INTRINSICS_PATH)
    # print(f"\n【相机内参信息】")
    # if intrinsics is not None:
    #     print(f"内参矩阵 K:\n{intrinsics}")
    # else:
    #     print("加载失败")

    # # 6. 加载OBJ模型（新增核心逻辑）
    # print(f"\n【OBJ模型信息】")
    # obj_mesh = None
    # obj_render_img = None
    # try:
    #     # 加载OBJ模型
    #     obj_mesh = load_obj_model(OBJECT_OBJ_PATH, device)
        
    #     # 打印模型关键信息
    #     verts = obj_mesh.verts_packed()
    #     faces = obj_mesh.faces_packed()
    #     textures = obj_mesh.textures
    #     print(f"模型路径: {OBJECT_OBJ_PATH}")
    #     print(f"顶点数: {verts.shape[0]} | 面数: {faces.shape[0]}")
    #     print(f"顶点范围: x({verts[:,0].min():.3f}~{verts[:,0].max():.3f}), y({verts[:,1].min():.3f}~{verts[:,1].max():.3f}), z({verts[:,2].min():.3f}~{verts[:,2].max():.3f})")
    #     print(f"是否有纹理: {'是' if textures is not None else '否（已使用默认颜色）'}")
        
    #     # 渲染OBJ模型为2D图像（用于可视化）
    #     obj_render_img = render_obj_mesh(obj_mesh, device)
    #     # 保存渲染图
    #     cv2.imwrite(os.path.join(SAVE_DIR, "obj_render.png"), cv2.cvtColor(obj_render_img, cv2.COLOR_RGB2BGR))
    #     print(f"✅ OBJ模型渲染图已保存到 {os.path.join(SAVE_DIR, 'obj_render.png')}")
        
    # except Exception as e:
    #     print(f"❌ 加载/渲染OBJ失败: {str(e)}")

    # # ---------------- 保存相机参数到文本文件 ----------------
    # param_file = os.path.join(SAVE_DIR, "camera_params.txt")
    # with open(param_file, "w") as f:
    #     f.write("=== 相机位姿 ===\n")
    #     f.write(f"位置 (x,y,z): {pose['location'] if (pose and pose['location']) else '加载失败'}\n")
    #     f.write(f"旋转 (pitch,yaw,roll): {pose['rotation'] if (pose and pose['rotation']) else '加载失败'}\n")
    #     f.write("\n=== 相机内参矩阵 K ===\n")
    #     if intrinsics is not None:
    #         f.write(np.array2string(intrinsics, precision=2))
    #     else:
    #         f.write("加载失败")
    # print(f"\n✅ 相机参数已保存到 {param_file}")

    # # ---------------- 保存OBJ模型信息到文本文件 ----------------
    # obj_info_file = os.path.join(SAVE_DIR, "obj_model_info.txt")
    # with open(obj_info_file, "w") as f:
    #     f.write("=== OBJ模型信息 ===\n")
    #     f.write(f"模型路径: {OBJECT_OBJ_PATH}\n")
    #     if obj_mesh is not None:
    #         verts = obj_mesh.verts_packed()
    #         faces = obj_mesh.faces_packed()
    #         f.write(f"顶点数: {verts.shape[0]}\n")
    #         f.write(f"面数: {faces.shape[0]}\n")
    #         f.write(f"顶点范围 - X: {verts[:,0].min():.3f} ~ {verts[:,0].max():.3f}\n")
    #         f.write(f"顶点范围 - Y: {verts[:,1].min():.3f} ~ {verts[:,1].max():.3f}\n")
    #         f.write(f"顶点范围 - Z: {verts[:,2].min():.3f} ~ {verts[:,2].max():.3f}\n")
    #         f.write(f"是否有纹理: {'是' if obj_mesh.textures is not None else '否'}\n")
    #     else:
    #         f.write("模型加载失败\n")
    # print(f"✅ OBJ模型信息已保存到 {obj_info_file}")

    # # ---------------- 可视化显示（新增OBJ渲染图） ----------------
    # print("\n" + "="*50)
    # print("开始可视化显示...（关闭窗口后程序结束）")
    # print("="*50)
    
    # # 调整子图布局（新增OBJ渲染图列）
    # plt.figure(figsize=(20, 5))
    
    # # 子图1：RGB
    # plt.subplot(1, 4, 1)
    # if rgb is not None:
    #     plt.imshow(rgb)
    #     plt.title("RGB Image")
    # else:
    #     plt.text(0.5, 0.5, "RGB加载失败", ha="center", va="center")
    # plt.axis("off")

    # # 子图2：深度图
    # plt.subplot(1, 4, 2)
    # if depth is not None:
    #     plt.imshow(depth, cmap="plasma")
    #     plt.title(f"Depth Map (min:{depth.min():.1f}m, max:{depth.max():.1f}m)")
    #     plt.colorbar(shrink=0.8)
    # else:
    #     plt.text(0.5, 0.5, "深度图加载失败", ha="center", va="center")
    # plt.axis("off")

    # # 子图3：掩码图
    # plt.subplot(1, 4, 3)
    # if mask is not None:
    #     plt.imshow(mask, cmap="gray")
    #     plt.title(f"Mask (pixels: {np.sum(mask):.0f})")
    # else:
    #     plt.text(0.5, 0.5, "掩码图加载失败", ha="center", va="center")
    # plt.axis("off")

    # # 子图4：OBJ模型渲染图（新增）
    # plt.subplot(1, 4, 4)
    # if obj_render_img is not None:
    #     plt.imshow(obj_render_img)
    #     plt.title(f"OBJ Model (verts: {obj_mesh.verts_packed().shape[0]:.0f})")
    # else:
    #     plt.text(0.5, 0.5, "OBJ加载/渲染失败", ha="center", va="center")
    # plt.axis("off")

    # plt.tight_layout()
    # # 保存组合可视化图
    # plt.savefig(os.path.join(SAVE_DIR, "debug_vis_with_obj.png"), dpi=150, bbox_inches="tight")
    # plt.show()

    # print("\n✅ 调试程序执行完成！")
    # print(f"所有结果已保存到: {os.path.abspath(SAVE_DIR)}")


# # # ================= 主程序（仅保留该块，无新增函数） =================
# # if __name__ == "__main__":
#     # ---------------- 配置区（修改为你的文件路径） ----------------
#     RGB_PATH = "/root/autodl-fs/data/data_test/location_000_random/rgb/random_000.png"       # RGB图路径
#     DEPTH_PATH = "/root/autodl-fs/data/data_test/location_000_random/depth/random_000.png"   # 深度图路径
#     MASK_PATH = "/root/autodl-fs/data/data_test/location_000_random/mask/random_000.png"     # 掩码图路径
#     POSE_PATH = "/root/autodl-fs/data/data_test/location_000_random/camera_pose/random_000.npz"   # 位姿文件路径
#     INTRINSICS_PATH = "/root/autodl-fs/data/data_test/location_000_random/camera_intrinsics/random_000.npz"  # 内参文件路径
#     SAVE_DIR = "./debug_results"  # 结果保存目录
#     object_path='/root/autodl-fs/data/object_model/TeslaModel3.obj'
#     # 创建保存目录
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # ---------------- 加载数据并打印信息 ----------------
#     print("="*50)
#     print("开始加载数据...")
#     print("="*50)

#     # 1. 加载RGB
#     rgb = load_rgb(RGB_PATH)
#     print(f"\n【RGB信息】")
#     if rgb is not None:
#         print(f"形状: {rgb.shape} | 数据类型: {rgb.dtype} | 像素值范围: {rgb.min()}~{rgb.max()}")
#         # 保存RGB图
#         rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(SAVE_DIR, "rgb.png"), rgb_bgr)
#         print(f"✅ RGB图已保存到 {os.path.join(SAVE_DIR, 'rgb.png')}")
#     else:
#         print("加载失败")

#     # 2. 加载深度
#     depth = load_depth(DEPTH_PATH)
#     print(f"\n【深度图信息】")
#     if depth is not None:
#         print(f"形状: {depth.shape} | 数据类型: {depth.dtype}")
#         print(f"深度范围: {depth.min():.2f} ~ {depth.max():.2f} 米")
#         # 保存深度可视化图+原始数据
#         depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
#         depth_norm = depth_norm.astype(np.uint8)
#         cv2.imwrite(os.path.join(SAVE_DIR, "depth_vis.png"), depth_norm)
#         np.save(os.path.join(SAVE_DIR, "depth_raw.npy"), depth)
#         print(f"✅ 深度图（可视化+原始数据）已保存到 {SAVE_DIR}")
#     else:
#         print("加载失败")

#     # 3. 加载掩码
#     mask = load_mask(MASK_PATH)
#     print(f"\n【掩码图信息】")
#     if mask is not None:
#         print(f"形状: {mask.shape} | 数据类型: {mask.dtype}")
#         print(f"最大蓝色区域像素数: {np.sum(mask)} | 取值范围: {mask.min()}~{mask.max()}")
#         # 保存掩码图
#         mask_vis = mask * 255
#         cv2.imwrite(os.path.join(SAVE_DIR, "mask.png"), mask_vis)
#         print(f"✅ 掩码图已保存到 {os.path.join(SAVE_DIR, 'mask.png')}")
#     else:
#         print("加载失败")

#     # 4. 加载相机位姿
#     pose = load_camera_pose(POSE_PATH)
#     print(f"\n【相机位姿信息】")
#     if pose is not None:
#         print(f"位置 (x,y,z): {[round(v, 3) for v in pose['location']]}")
#         print(f"旋转 (pitch,yaw,roll): {[round(v, 3) for v in pose['rotation']]}")
#     else:
#         print("加载失败")

#     # 5. 加载相机内参
#     intrinsics = load_camera_intrinsics(INTRINSICS_PATH)
#     print(f"\n【相机内参信息】")
#     if intrinsics is not None:
#         print(f"内参矩阵 K:\n{intrinsics}")
#     else:
#         print("加载失败")

#     # ---------------- 保存相机参数到文本文件 ----------------
#     param_file = os.path.join(SAVE_DIR, "camera_params.txt")
#     with open(param_file, "w") as f:
#         f.write("=== 相机位姿 ===\n")
#         f.write(f"位置 (x,y,z): {pose if pose else '加载失败'}\n")
#         f.write(f"旋转 (pitch,yaw,roll): {pose['rotation'] if pose else '加载失败'}\n")
#         f.write("\n=== 相机内参矩阵 K ===\n")
#         if intrinsics is not None:
#             f.write(np.array2string(intrinsics, precision=2))
#         else:
#             f.write("加载失败")
#     print(f"\n✅ 相机参数已保存到 {param_file}")

#     # ---------------- 可视化显示 ----------------
#     print("\n" + "="*50)
#     print("开始可视化显示...（关闭窗口后程序结束）")
#     print("="*50)
    
#     # 创建子图
#     plt.figure(figsize=(15, 5))
    
#     # 子图1：RGB
#     plt.subplot(1, 3, 1)
#     if rgb is not None:
#         plt.imshow(rgb)
#         plt.title("RGB Image")
#     else:
#         plt.text(0.5, 0.5, "RGB加载失败", ha="center", va="center")
#     plt.axis("off")

#     # 子图2：深度图
#     plt.subplot(1, 3, 2)
#     if depth is not None:
#         plt.imshow(depth, cmap="plasma")
#         plt.title(f"Depth Map (min:{depth.min():.1f}m, max:{depth.max():.1f}m)")
#         plt.colorbar(shrink=0.8)
#     else:
#         plt.text(0.5, 0.5, "深度图加载失败", ha="center", va="center")
#     plt.axis("off")

#     # 子图3：掩码图
#     plt.subplot(1, 3, 3)
#     if mask is not None:
#         plt.imshow(mask, cmap="gray")
#         plt.title(f"Mask (max blue area: {np.sum(mask)} pixels)")
#     else:
#         plt.text(0.5, 0.5, "掩码图加载失败", ha="center", va="center")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()
#     # 保存图片
#     plt.savefig(os.path.join(SAVE_DIR, "debug_vis.png"))

#     print("\n✅ 调试程序执行完成！")





