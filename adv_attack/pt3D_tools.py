import math

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
    PerspectiveCameras,
    TexturesUV,TexturesVertex, TexturesAtlas)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
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
    image_tensor_temp = image_tensor.clone().detach()
    image_np = image_tensor_temp.cpu().clamp(0.0, 1.0).numpy()  # 限制范围，避免异常值
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
        plt.close('all')  
        # print(f"渲染结果已保存至：{save_path}")



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



def load_obj_model(obj_path: str, device: torch.device):
    """
    加载 OBJ 并按材质渲染：
    - 有贴图 → 使用 UV 纹理
    - 无贴图 → 使用材料颜色创建纯色UV纹理（修复维度匹配问题）
    """
    print(f"Loading OBJ: {obj_path}")

    verts, faces, aux = load_obj(obj_path, load_textures=True)
    verts = verts.to(device)
    faces_idx = faces.verts_idx.to(device)

    has_uv = aux.verts_uvs is not None and faces.textures_idx is not None and len(aux.verts_uvs) > 0
    print("OBJ检测:")
    print("verts_uvs:", None if aux.verts_uvs is None else aux.verts_uvs.shape)
    print("faces_uvs:", None if faces.textures_idx is None else faces.textures_idx.shape)
    print("texture_images:", aux.texture_images)
    print("materials:", list(aux.material_colors.keys()) if aux.material_colors else None)

    # faces.materials_idx 对应的整数索引，需要映射到 aux.material_colors
    material_names = list(aux.material_colors.keys()) if aux.material_colors else []

    meshes_list = []

    for mat_idx in faces.materials_idx.unique().tolist():
        # 对应材质名字
        mat_name = material_names[mat_idx] if mat_idx < len(material_names) else None

        # 找到使用这个材质的面
        face_mask = (faces.materials_idx == mat_idx)
        face_indices = face_mask.nonzero(as_tuple=True)[0]
        
        # 提取当前材质对应的面和UV索引（核心修复：仅保留当前材质的索引）
        current_faces_idx = faces_idx[face_indices]  # 当前材质的面索引
        current_faces_uvs = faces.textures_idx[face_indices].to(device) if has_uv else None
        current_verts_uvs = aux.verts_uvs.to(device) if has_uv else None

        # 是否有纹理图片
        if has_uv and aux.texture_images is not None and mat_name in aux.texture_images:
            # 有纹理图：使用当前材质的UV索引
            tex_img = aux.texture_images[mat_name].to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            texture_image = tex_img.permute(0, 2, 3, 1).contiguous()
            tex = TexturesUV(
                maps=texture_image,
                faces_uvs=current_faces_uvs[None],  # 仅当前材质的面UV索引
                verts_uvs=current_verts_uvs[None]    # 全局UV（但索引仅指向当前材质的面）
            )
        # 无纹理图片但有UV → 创建纯色UV纹理
        elif has_uv:
            # 获取材质漫反射颜色
            try:
                diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
                if isinstance(diffuse_color, torch.Tensor):
                    color = diffuse_color.to(device=device, dtype=torch.float32).detach()
                else:
                    # 处理颜色是列表/数组的情况，确保维度为3
                    color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
                    if color.ndim == 2:  # 修复颜色维度异常（如[[r,g,b],[r,g,b]]）
                        color = color[0]
            except (KeyError, TypeError, IndexError):
                color = torch.tensor([0.7, 0.7, 0.7], device=device)  # 默认灰色
        
            # 确保颜色是1维张量（RGB）
            color = color.squeeze()
            if color.numel() != 3:
                color = torch.tensor([0.7, 0.7, 0.7], device=device)
        
            # 创建纯色UV纹理图（512x512，适配UV映射）
            texture_image = color.expand(1, 512, 512, 3).contiguous()
            
            # 用纯色纹理创建UV纹理（使用当前材质的UV索引）
            tex = TexturesUV(
                maps=texture_image,
                faces_uvs=current_faces_uvs[None],  # 核心：仅当前材质的面UV索引
                verts_uvs=current_verts_uvs[None]
            )
        # 无UV也无纹理 → 降级使用顶点颜色
        else:
            try:
                diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
                if isinstance(diffuse_color, torch.Tensor):
                    color = diffuse_color.to(device=device, dtype=torch.float32).detach()
                else:
                    color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
                    if color.ndim == 2:
                        color = color[0]
            except (KeyError, TypeError, IndexError):
                color = torch.tensor([0.7, 0.7, 0.7], device=device)

            color = color.squeeze()
            if color.numel() != 3:
                color = torch.tensor([0.7, 0.7, 0.7], device=device)

            verts_color = color.expand(len(verts), 3)
            tex = TexturesVertex(verts_features=verts_color[None])
        
        # 创建当前材质的Mesh（使用当前材质的面索引）
        mesh = Meshes(
            verts=[verts],  # 全局顶点（OBJ的所有顶点）
            faces=[current_faces_idx],  # 仅当前材质的面
            textures=tex
        )
        meshes_list.append(mesh)



    print("\n" + "="*60)
    print("📦 模型尺寸 & 原点信息")
    print("="*60)

    # 1. 顶点范围
    v_min = verts.min(dim=0)[0]
    v_max = verts.max(dim=0)[0]
    print(f"顶点最小坐标 (min): {v_min.tolist()}")
    print(f"顶点最大坐标 (max): {v_max.tolist()}")

    # 2. 模型尺寸 (width, height, depth)
    size = v_max - v_min
    print(f"模型尺寸 X(宽): {size[0]:.3f}")
    print(f"模型尺寸 Y(深): {size[1]:.3f}")
    print(f"模型尺寸 Z(高): {size[2]:.3f}")

    # 3. 模型几何中心
    center = (v_min + v_max) / 2
    print(f"模型几何中心: {center.tolist()}")

    # 4. 模型底面中心（你CARLA对齐用的地面点）
    bottom_center = torch.tensor([
        center[0],
        center[1],
        v_min[2]  # Z取最低 = 地面
    ], device=device)
    print(f"模型底面中心点 (地面接触点): {bottom_center.tolist()}")

    # 5. 当前模型原点（就是OBJ导出时的原点：(0,0,0) 在模型中的位置）
    print(f"OBJ文件原点 (0,0,0) 相对于模型的位置: [0, 0, 0]")
    print("⚠️  说明：PyTorch3D加载后的原点 = Blender里设置的原点")
    print("="*60 + "\n")

    print(f"共生成 {len(meshes_list)} 个子 Mesh (材质分割)")

    return meshes_list

# def load_obj_model_return_mesh_material(obj_path: str, device: torch.device):
#     """
#     加载 OBJ 并按材质渲染：
#     - 有贴图 → 使用 UV 纹理
#     - 无贴图 → 使用材料颜色创建纯色UV纹理（修复维度匹配问题）
#     """
#     print(f"Loading OBJ: {obj_path}")

#     verts, faces, aux = load_obj(obj_path, load_textures=True)
#     verts = verts.to(device)
#     faces_idx = faces.verts_idx.to(device)

#     has_uv = aux.verts_uvs is not None and faces.textures_idx is not None and len(aux.verts_uvs) > 0
#     print("OBJ检测:")
#     print("verts_uvs:", None if aux.verts_uvs is None else aux.verts_uvs.shape)
#     print("faces_uvs:", None if faces.textures_idx is None else faces.textures_idx.shape)
#     print("texture_images:", aux.texture_images)
#     print("materials:", list(aux.material_colors.keys()) if aux.material_colors else None)

#     # faces.materials_idx 对应的整数索引，需要映射到 aux.material_colors
#     material_names = list(aux.material_colors.keys()) if aux.material_colors else []

#     meshes_list = []
#     material_names_list = []

#     for mat_idx in faces.materials_idx.unique().tolist():
#         # 对应材质名字
#         mat_name = material_names[mat_idx] if mat_idx < len(material_names) else None

#         # 找到使用这个材质的面
#         face_mask = (faces.materials_idx == mat_idx)
#         face_indices = face_mask.nonzero(as_tuple=True)[0]
        
#         # 提取当前材质对应的面和UV索引（核心修复：仅保留当前材质的索引）
#         current_faces_idx = faces_idx[face_indices]  # 当前材质的面索引
#         current_faces_uvs = faces.textures_idx[face_indices].to(device) if has_uv else None
#         current_verts_uvs = aux.verts_uvs.to(device) if has_uv else None

#         # 是否有纹理图片
#         if has_uv and aux.texture_images is not None and mat_name in aux.texture_images:
#             # 有纹理图：使用当前材质的UV索引
#             tex_img = aux.texture_images[mat_name].to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#             texture_image = tex_img.permute(0, 2, 3, 1).contiguous()
#             tex = TexturesUV(
#                 maps=texture_image,
#                 faces_uvs=current_faces_uvs[None],  # 仅当前材质的面UV索引
#                 verts_uvs=current_verts_uvs[None]    # 全局UV（但索引仅指向当前材质的面）
#             )
#         # 无纹理图片但有UV → 创建纯色UV纹理
#         elif has_uv:
#             # 获取材质漫反射颜色
#             try:
#                 diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
#                 if isinstance(diffuse_color, torch.Tensor):
#                     color = diffuse_color.to(device=device, dtype=torch.float32).detach()
#                 else:
#                     # 处理颜色是列表/数组的情况，确保维度为3
#                     color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
#                     if color.ndim == 2:  # 修复颜色维度异常（如[[r,g,b],[r,g,b]]）
#                         color = color[0]
#             except (KeyError, TypeError, IndexError):
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)  # 默认灰色
        
#             # 确保颜色是1维张量（RGB）
#             color = color.squeeze()
#             if color.numel() != 3:
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)
        
#             # 创建纯色UV纹理图（512x512，适配UV映射）
#             texture_image = color.expand(1, 512, 512, 3).contiguous()
            
#             # 用纯色纹理创建UV纹理（使用当前材质的UV索引）
#             tex = TexturesUV(
#                 maps=texture_image,
#                 faces_uvs=current_faces_uvs[None],  # 核心：仅当前材质的面UV索引
#                 verts_uvs=current_verts_uvs[None]
#             )
#         # 无UV也无纹理 → 降级使用顶点颜色
#         else:
#             try:
#                 diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
#                 if isinstance(diffuse_color, torch.Tensor):
#                     color = diffuse_color.to(device=device, dtype=torch.float32).detach()
#                 else:
#                     color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
#                     if color.ndim == 2:
#                         color = color[0]
#             except (KeyError, TypeError, IndexError):
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)

#             color = color.squeeze()
#             if color.numel() != 3:
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)

#             verts_color = color.expand(len(verts), 3)
#             tex = TexturesVertex(verts_features=verts_color[None])
        
#         # 创建当前材质的Mesh（使用当前材质的面索引）
#         mesh = Meshes(
#             verts=[verts],  # 全局顶点（OBJ的所有顶点）
#             faces=[current_faces_idx],  # 仅当前材质的面
#             textures=tex
#         )
#         meshes_list.append(mesh)
#         material_names_list.append(mat_name)

#     print(f"共生成 {len(meshes_list)} 个子 Mesh (材质分割)")
#     return meshes_list,material_names_list

# def load_obj_model_return_mesh_material(obj_path: str, device: torch.device):
#     """
#     加载 OBJ 并按材质渲染：
#     - 有贴图 → 使用 UV 纹理
#     - 无贴图 → 使用材料颜色创建纯色UV纹理
#     内置透明度 d 处理，保持 RGB 3通道，兼容原有 Shader
#     """
#     print(f"Loading OBJ: {obj_path}")

#     verts, faces, aux = load_obj(obj_path, load_textures=True)
#     verts = verts.to(device)
#     faces_idx = faces.verts_idx.to(device)

#     has_uv = aux.verts_uvs is not None and faces.textures_idx is not None and len(aux.verts_uvs) > 0
#     print("OBJ检测:")
#     print("verts_uvs:", None if aux.verts_uvs is None else aux.verts_uvs.shape)
#     print("faces_uvs:", None if faces.textures_idx is None else faces.textures_idx.shape)
#     print("texture_images:", aux.texture_images)
#     print("materials:", list(aux.material_colors.keys()) if aux.material_colors else None)

#     material_names = list(aux.material_colors.keys()) if aux.material_colors else []

#     # 解析 mtl 获取透明度 d
#     def parse_mtl_alpha(mtl_path):
#         mat_alpha = {}
#         current_mat = None
#         if not os.path.exists(mtl_path):
#             return mat_alpha
#         with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = line.split()
#                 if parts[0] == 'newmtl':
#                     current_mat = parts[1]
#                     mat_alpha[current_mat] = 1.0
#                 elif parts[0] == 'd' and current_mat:
#                     try:
#                         mat_alpha[current_mat] = float(parts[1])
#                     except:
#                         mat_alpha[current_mat] = 1.0
#         return mat_alpha

#     obj_dir = os.path.dirname(obj_path)
#     mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
#     mat_alpha = parse_mtl_alpha(mtl_path)

#     meshes_list = []
#     material_names_list = []

#     for mat_idx in faces.materials_idx.unique().tolist():
#         mat_name = material_names[mat_idx] if mat_idx < len(material_names) else None
#         face_mask = (faces.materials_idx == mat_idx)
#         face_indices = face_mask.nonzero(as_tuple=True)[0]

#         current_faces_idx = faces_idx[face_indices]
#         current_faces_uvs = faces.textures_idx[face_indices].to(device) if has_uv else None
#         current_verts_uvs = aux.verts_uvs.to(device) if has_uv else None

#         # 读取当前材质透明度
#         alpha = mat_alpha.get(mat_name, 1.0)

#         texture_image = None
#         tex = None

#         # ==============================================
#         # 有纹理图：加载后直接乘以透明度（保持 RGB 3 通道）
#         # ==============================================
#         if has_uv and aux.texture_images is not None and mat_name in aux.texture_images:
#             tex_img = aux.texture_images[mat_name].to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#             texture_image = tex_img.permute(0, 2, 3, 1).contiguous()
#             # 透明度直接乘到颜色上
#             texture_image = texture_image * alpha

#             tex = TexturesUV(
#                 maps=texture_image,
#                 faces_uvs=current_faces_uvs[None],
#                 verts_uvs=current_verts_uvs[None]
#             )

#         # ==============================================
#         # 无纹理有UV：纯色 * 透明度
#         # ==============================================
#         elif has_uv:
#             try:
#                 diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
#                 if isinstance(diffuse_color, torch.Tensor):
#                     color = diffuse_color.to(device, torch.float32).detach()
#                 else:
#                     color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
#                     if color.ndim == 2:
#                         color = color[0]
#             except (KeyError, TypeError, IndexError):
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)

#             color = color.squeeze()
#             if color.numel() != 3:
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)

#             # 纯色 * 透明度
#             color = color * alpha
#             texture_image = color.expand(1, 512, 512, 3).contiguous()

#             tex = TexturesUV(
#                 maps=texture_image,
#                 faces_uvs=current_faces_uvs[None],
#                 verts_uvs=current_verts_uvs[None]
#             )

#         # ==============================================
#         # 无UV：顶点色 * 透明度
#         # ==============================================
#         else:
#             try:
#                 diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
#                 if isinstance(diffuse_color, torch.Tensor):
#                     color = diffuse_color.to(device, torch.float32).detach()
#                 else:
#                     color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
#                     if color.ndim == 2:
#                         color = color[0]
#             except (KeyError, TypeError, IndexError):
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)

#             color = color.squeeze()
#             if color.numel() != 3:
#                 color = torch.tensor([0.7, 0.7, 0.7], device=device)

#             # 颜色 * 透明度
#             color = color * alpha
#             verts_color = color.expand(len(verts), 3).contiguous()
#             tex = TexturesVertex(verts_features=verts_color[None])

#         # 构建 mesh
#         mesh = Meshes(
#             verts=[verts],
#             faces=[current_faces_idx],
#             textures=tex
#         )
#         meshes_list.append(mesh)
#         material_names_list.append(mat_name)

#     print(f"共生成 {len(meshes_list)} 个子 Mesh (材质分割)")
#     return meshes_list, material_names_list

def parse_mtl(mtl_path):
    materials = {}
    current_mat = None

    if not os.path.exists(mtl_path):
        return materials

    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("newmtl"):
                current_mat = line.split()[1]
                materials[current_mat] = {}
            elif current_mat is None:
                continue
            else:
                tokens = line.split()
                if len(tokens) == 0:
                    continue

                key = tokens[0]

                if key in ["map_Kd", "map_d"]:
                    materials[current_mat][key] = tokens[-1]

                elif key in ["d", "Tr"]:
                    materials[current_mat]["alpha"] = float(tokens[1])

    return materials


# =========================
# 2. texture + alpha bake
# =========================
def load_texture_baked_alpha(map_kd_path, map_d_path, alpha_scalar, device):
    rgb = Image.open(map_kd_path).convert("RGB")
    rgb = np.array(rgb).astype(np.float32) / 255.0

    # alpha map
    if map_d_path is not None and os.path.exists(map_d_path):
        alpha = Image.open(map_d_path).convert("L")
        alpha = np.array(alpha).astype(np.float32) / 255.0
    else:
        alpha = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    # scalar alpha（来自 d / Tr）
    if alpha_scalar is not None:
        alpha = alpha * alpha_scalar

    # ===== bake =====
    eps = 1e-6
    rgb = rgb * alpha[..., None] + (1 - alpha[..., None]) * eps

    tex = torch.from_numpy(rgb).to(device).unsqueeze(0)  # (1,H,W,3)
    return tex


# =========================
# 3. 主函数
# =========================
def load_obj_model_return_mesh_material(obj_path: str, device: torch.device,scale=1):

    print(f"Loading OBJ: {obj_path}")

    verts, faces, aux = load_obj(obj_path, load_textures=True)
    verts = verts/scale
    verts = verts.to(device)
    faces_idx = faces.verts_idx.to(device)

    has_uv = (
        aux.verts_uvs is not None
        and faces.textures_idx is not None
        and len(aux.verts_uvs) > 0
    )

    print("OBJ检测:")
    print("verts_uvs:", None if aux.verts_uvs is None else aux.verts_uvs.shape)
    print("faces_uvs:", None if faces.textures_idx is None else faces.textures_idx.shape)
    print("materials:", list(aux.material_colors.keys()) if aux.material_colors else None)

    # ===== 解析 MTL =====
    mtl_path = obj_path.replace(".obj", ".mtl")
    mtl_data = parse_mtl(mtl_path)

    material_names = list(aux.material_colors.keys()) if aux.material_colors else []

    meshes_list = []
    material_names_list = []

    for mat_idx in faces.materials_idx.unique().tolist():

        mat_name = material_names[mat_idx] if mat_idx < len(material_names) else None

        face_mask = (faces.materials_idx == mat_idx)
        face_indices = face_mask.nonzero(as_tuple=True)[0]

        current_faces_idx = faces_idx[face_indices]

        current_faces_uvs = (
            faces.textures_idx[face_indices].to(device) if has_uv else None
        )

        current_verts_uvs = aux.verts_uvs.to(device) if has_uv else None

        # =========================
        # 有 UV：优先走 texture
        # =========================
        if has_uv:

            mtl_info = mtl_data.get(mat_name, {})

            map_kd = mtl_info.get("map_Kd", None)
            map_d = mtl_info.get("map_d", None)
            alpha_scalar = mtl_info.get("alpha", 1.0)

            if map_kd is not None:
                map_kd = os.path.join(os.path.dirname(obj_path), map_kd)

            if map_d is not None:
                map_d = os.path.join(os.path.dirname(obj_path), map_d)

            # ===== 有贴图 =====
            if map_kd is not None and os.path.exists(map_kd):

                texture_image = load_texture_baked_alpha(
                    map_kd,
                    map_d,
                    alpha_scalar,
                    device
                )

                tex = TexturesUV(
                    maps=texture_image,
                    faces_uvs=current_faces_uvs[None],
                    verts_uvs=current_verts_uvs[None]
                )

            # ===== 无贴图：纯色 =====
            else:
                try:
                    diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
                    color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
                    if color.ndim == 2:
                        color = color[0]
                except:
                    color = torch.tensor([0.7, 0.7, 0.7], device=device)

                color = color.clamp(0, 1)

                # alpha 作用到颜色
                color = color * alpha_scalar

                texture_image = color.view(1, 1, 1, 3).expand(1, 512, 512, 3).contiguous()

                tex = TexturesUV(
                    maps=texture_image,
                    faces_uvs=current_faces_uvs[None],
                    verts_uvs=current_verts_uvs[None]
                )

        # =========================
        # 无 UV：vertex color
        # =========================
        else:
            try:
                diffuse_color = aux.material_colors[mat_name]["diffuse_color"]
                color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)
                if color.ndim == 2:
                    color = color[0]
            except:
                color = torch.tensor([0.7, 0.7, 0.7], device=device)

            color = color.clamp(0, 1)

            verts_color = color.expand(len(verts), 3)

            tex = TexturesVertex(verts_features=verts_color[None])

        mesh = Meshes(
            verts=[verts],
            faces=[current_faces_idx],
            textures=tex
        )

        meshes_list.append(mesh)
        material_names_list.append(mat_name)

###########################################################################
    # ===================== 【自动计算：模型原点 + 大小】 =====================
    ###########################################################################
    print("\n" + "="*60)
    print("📦 模型尺寸 & 原点信息")
    print("="*60)

    # 1. 顶点范围
    v_min = verts.min(dim=0)[0]
    v_max = verts.max(dim=0)[0]
    print(f"顶点最小坐标 (min): {v_min.tolist()}")
    print(f"顶点最大坐标 (max): {v_max.tolist()}")

    # 2. 模型尺寸 (width, height, depth)
    size = v_max - v_min
    print(f"模型尺寸 X(宽): {size[0]:.3f}")
    print(f"模型尺寸 Y(深): {size[1]:.3f}")
    print(f"模型尺寸 Z(高): {size[2]:.3f}")

    # 3. 模型几何中心
    center = (v_min + v_max) / 2
    print(f"模型几何中心: {center.tolist()}")

    # 4. 模型底面中心（你CARLA对齐用的地面点）
    bottom_center = torch.tensor([
        center[0],
        center[1],
        v_min[2]  # Z取最低 = 地面
    ], device=device)
    print(f"模型底面中心点 (地面接触点): {bottom_center.tolist()}")

    # 5. 当前模型原点（就是OBJ导出时的原点：(0,0,0) 在模型中的位置）
    print(f"OBJ文件原点 (0,0,0) 相对于模型的位置: [0, 0, 0]")
    print("⚠️  说明：PyTorch3D加载后的原点 = Blender里设置的原点")
    print("="*60 + "\n")

    print(f"共生成 {len(meshes_list)} 个子 Mesh (材质分割)")

    return meshes_list, material_names_list

def generate_camera_from_params(
    pose_paths: list,  
    device: torch.device,
    fov: float = 110.0,  
    img_size: tuple = (720, 1280)  # ( height,width)
) -> FoVPerspectiveCameras:  # 修正返回类型：PerspectiveCameras → FoVPerspectiveCameras
    """
    批量加载位姿文件，生成单个批量相机对象（而非列表）
    参数：
        pose_paths: 相机位姿文件路径列表
        device: 运行设备 (cpu/cuda)
        fov: 相机视场角（单位：度），默认110°
        img_size: 图像尺寸 (width, height)，默认(720,1280)
    返回：
        FoVPerspectiveCameras: 批量相机对象（包含len(pose_paths)个相机）
    """
    # ================= 1 初始化存储列表 =================
    R_list = []  # 存储所有相机的旋转矩阵 (3,3)
    T_list = []  # 存储所有相机的平移向量 (3,)
    H,W=img_size
    fov_horizontal_deg = fov
    aspect = W / H
    fov_h_rad = math.radians(fov_horizontal_deg)
    fov_v_rad = math.degrees(2 * math.atan(math.tan(fov_h_rad / 2) / aspect))
    # ================= 2 批量加载位姿并计算外参 =================
    for pose_path in pose_paths:
        # 加载单个位姿文件
        pose = load_camera_pose(pose_path)
        if pose is None:
            raise ValueError(f"位姿文件加载失败：{pose_path}")
        
        # 提取相机位置,#可能要取负号,以calra为准，相机相对车辆原点的坐标（车辆底部的中心）
        x, y, z = pose['location']
        
        # 计算俯仰角和方位角（修正坐标系转换）
        distance=np.sqrt(x**2 + z**2 + y**2)
        pitch_carla = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        pitch=pitch_carla
        # 修正水平角度
        yaw_crla = -np.degrees(np.arctan2(y, x))  
        yaw=(yaw_crla-90)+180 # 修正坐标系转换，需要注意车辆坐标，carla坐标，opytorch3D坐标
        # 生成单个相机的外参
        R_single, T_single = look_at_view_transform(
            dist=distance,
            elev=pitch,
            azim=yaw
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
        fov=fov_v_rad,                    # 所有相机共用的视场角
        znear=0.1,                 # 所有相机共用的近裁剪面
        zfar=100.0,                 # 所有相机共用的远裁剪面
        # aspect_ratio=img_size[1]/img_size[0],  # 宽高比
        # 显式设置图像尺寸，确保和渲染配置匹配
        
    )

    return batch_cameras


# def generate_camera_from_params_v2(
#     cam_relative_pos,
#     cam_relative_rot,  
#     device: torch.device,
#     fov: float = 110.0,  
#     img_size: tuple = (720, 1280)  # ( height,width)
# ) -> FoVPerspectiveCameras:  # 修正返回类型：PerspectiveCameras → FoVPerspectiveCameras
#     """
#     批量加载位姿文件，生成单个批量相机对象（而非列表）
#     参数：
#         pose_paths: 相机位姿文件路径列表
#         device: 运行设备 (cpu/cuda)
#         fov: 相机视场角（单位：度），默认110°
#         img_size: 图像尺寸 (width, height)，默认(720,1280)
#     返回：
#         FoVPerspectiveCameras: 批量相机对象（包含len(pose_paths)个相机）
#     """
#     # ================= 1 初始化存储列表 =================
#     R_list = []  # 存储所有相机的旋转矩阵 (3,3)
#     T_list = []  # 存储所有相机的平移向量 (3,)
#     H,W=img_size
#     fov_horizontal_deg = fov
#     aspect = W / H
#     fov_h_rad = math.radians(fov_horizontal_deg)
#     fov_v_rad = math.degrees(2 * math.atan(math.tan(fov_h_rad / 2) / aspect))
#     # ================= 2 批量加载位姿并计算外参 =================
#     for pose_path in pose_paths:
#         # 加载单个位姿文件
#         pose = load_camera_pose(pose_path)
#         if pose is None:
#             raise ValueError(f"位姿文件加载失败：{pose_path}")
        
#         # 提取相机位置,#可能要取负号,以calra为准，相机相对车辆原点的坐标（车辆底部的中心）
#         x, y, z = pose['location']
        
#         # 计算俯仰角和方位角（修正坐标系转换）
#         distance=np.sqrt(x**2 + z**2 + y**2)
#         pitch_carla = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
#         pitch=pitch_carla
#         # 修正水平角度
#         yaw_crla = -np.degrees(np.arctan2(y, x))  
#         yaw=(yaw_crla-90)+180 # 修正坐标系转换，需要注意车辆坐标，carla坐标，opytorch3D坐标
#         # 生成单个相机的外参
#         R_single, T_single = look_at_view_transform(
#             dist=distance,
#             elev=pitch,
#             azim=yaw
#         )
        
#         # 转换为tensor并移到指定设备，去除batch维度
#         R_single = R_single.squeeze(0).to(device)  # (3,3)
#         T_single = T_single.squeeze(0).to(device)  # (3,)
        
#         # 添加到列表
#         R_list.append(R_single)
#         T_list.append(T_single)

#     # ================= 3 拼接为批量外参 =================
#     # 拼接为 (B, 3, 3) 旋转矩阵（B=相机数量）
#     R_batch = torch.stack(R_list, dim=0)
#     # 拼接为 (B, 3) 平移向量
#     T_batch = torch.stack(T_list, dim=0)

#     # ================= 4 生成批量相机对象 =================
#     batch_cameras = FoVPerspectiveCameras(
#         device=device,
#         R=R_batch,                  # 批量旋转矩阵 (B, 3, 3)
#         T=T_batch,                  # 批量平移向量 (B, 3)
#         fov=fov_v_rad,                    # 所有相机共用的视场角
#         znear=0.1,                 # 所有相机共用的近裁剪面
#         zfar=100.0,                 # 所有相机共用的远裁剪面
#         # aspect_ratio=img_size[1]/img_size[0],  # 宽高比
#         # 显式设置图像尺寸，确保和渲染配置匹配
        
#     )

#     return batch_cameras




# def generate_camera_from_params_v2(
#     cam_relative_pos,   # (B,3) or (3,) -> [x, y, z]
#     cam_relative_rot,   # (B,3) or (3,) -> [pitch, yaw, roll] (deg)
#     device: torch.device,
#     fov: float = 110.0,
#     img_size: tuple = (720, 1280)
# ) -> FoVPerspectiveCameras:

#     # ===================== 图像参数 =====================
#     H, W = img_size
#     aspect = W / H

#     # 水平FOV -> 垂直FOV
#     fov_h_rad = math.radians(fov)
#     fov_v_rad = math.degrees(
#         2 * math.atan(math.tan(fov_h_rad / 2) / aspect)
#     )

#     # ===================== batch 统一 =====================
#     if cam_relative_pos.ndim == 1:
#         cam_relative_pos = cam_relative_pos.unsqueeze(0)
#     if cam_relative_rot.ndim == 1:
#         cam_relative_rot = cam_relative_rot.unsqueeze(0)

#     cam_relative_pos = cam_relative_pos.to(device)
#     cam_relative_rot = cam_relative_rot.to(device)

#     B = cam_relative_pos.shape[0]

#     R_list = []
#     T_list = []

#     # ===================== 构造外参 =====================
#     for i in range(B):

#         C = cam_relative_pos[i]  # 相机中心 (x,y,z)

#         pitch_deg, yaw_deg, roll_deg = cam_relative_rot[i]

#         # 角度 -> 弧度
#         pitch = torch.deg2rad(-pitch_deg)
#         yaw   = torch.deg2rad(yaw_deg)
#         roll  = torch.deg2rad(roll_deg)

#         # ===================== 构造旋转矩阵 =====================
#         # pitch -> X轴
#         Rx = torch.stack([
#             torch.stack([torch.tensor(1.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)]),
#             torch.stack([torch.tensor(0.0, device=device), torch.cos(pitch), -torch.sin(pitch)]),
#             torch.stack([torch.tensor(0.0, device=device), torch.sin(pitch),  torch.cos(pitch)])
#         ])

#         # yaw -> Y轴
#         Ry = torch.stack([
#             torch.stack([ torch.cos(yaw), torch.tensor(0.0, device=device), torch.sin(yaw)]),
#             torch.stack([ torch.tensor(0.0, device=device), torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)]),
#             torch.stack([-torch.sin(yaw), torch.tensor(0.0, device=device), torch.cos(yaw)])
#         ])

#         # roll -> Z轴
#         Rz = torch.stack([
#             torch.stack([torch.cos(roll), -torch.sin(roll), torch.tensor(0.0, device=device)]),
#             torch.stack([torch.sin(roll),  torch.cos(roll), torch.tensor(0.0, device=device)]),
#             torch.stack([torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)])
#         ])

#         # ===================== XYZ 顺序 =====================
#         R = Rx @ Ry @ Rz

#         # ===================== 外参 =====================
#         T = -R @ C

#         R_list.append(R)
#         T_list.append(T)

#     R_batch = torch.stack(R_list, dim=0)   # (B,3,3)
#     T_batch = torch.stack(T_list, dim=0)   # (B,3)

#     # ===================== 正交化（防数值误差） =====================
#     try:
#         U, _, V = torch.linalg.svd(R_batch)
#         R_batch = torch.bmm(U, V.transpose(1, 2))
#     except:
#         pass

#     # ===================== 构建相机 =====================
#     cameras = FoVPerspectiveCameras(
#         device=device,
#         R=R_batch,
#         T=T_batch,
#         fov=fov_v_rad,
#         znear=0.1,
#         zfar=100.0,
#         aspect_ratio=aspect,
#     )

#     return cameras


def generate_camera_from_params_v2(
    cam_relative_pos,   # (B,3) or (3,)
    cam_relative_rot,   # 已不再使用（保留接口）
    device: torch.device,
    fov: float = 90,
    img_size: tuple = (512, 512)
) -> FoVPerspectiveCameras:

    # ===================== 图像参数 =====================
    H, W = img_size
    aspect = W / H

    fov_h_rad = math.radians(fov)
    fov_v_rad = math.degrees(
        2 * math.atan(math.tan(fov_h_rad / 2) / aspect)
    )

    # ===================== batch 统一 =====================
    if cam_relative_pos.ndim == 1:
        cam_relative_pos = cam_relative_pos.unsqueeze(0)

    cam_relative_pos = cam_relative_pos.to(device)

    B = cam_relative_pos.shape[0]

    R_list = []
    T_list = []

    # ===================== look-at 构造 =====================
    for i in range(B):

        x, y, z = cam_relative_pos[i]
        
        x = x.detach().cpu().item()
        y = y.detach().cpu().item()
        z = z.detach().cpu().item()

        # 计算俯仰角和方位角（修正坐标系转换）
        distance=np.sqrt(x**2 + z**2 + y**2)
        pitch_carla = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        pitch=pitch_carla
        # 修正水平角度
        yaw = 180-np.degrees(np.arctan2(x, y))
        print(yaw)
        # yaw_crla = -np.degrees(np.arctan2(y, x))  
        # yaw=(yaw-90)+180 # 修正坐标系转换，需要注意车辆坐标，carla坐标，opytorch3D坐标
        # 生成单个相机的外参
        R_single, T_single = look_at_view_transform(
            dist=distance,
            elev=pitch,
            azim=yaw
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
        fov=fov_v_rad,                    # 所有相机共用的视场角
        znear=0.1,                 # 所有相机共用的近裁剪面
        zfar=100.0,                 # 所有相机共用的远裁剪面
        # aspect_ratio=img_size[1]/img_size[0],  # 宽高比
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




def light_set_fixed(
    device,
    ambient_color=((1, 1, 1),),    # 官方默认环境光颜色
    diffuse_color=((0.3, 0.3, 0.3),),    # 官方默认漫反射光颜色
    specular_color=((0.2, 0.2, 0.2),),   # 官方默认镜面反射光颜色
    location=((2, 4, 2),)                # 官方默认光源位置（可覆盖为你的原位置）
):
    """
    配置固定点光源
    :param device: 设备 (str/torch.device)，如 "cuda" / "cpu"
    :param ambient_color: 环境光RGB颜色，格式 ((r,g,b),) 或 [[r,g,b]]
    :param diffuse_color: 漫反射光RGB颜色，格式 ((r,g,b),) 或 [[r,g,b]]
    :param specular_color: 镜面反射光RGB颜色，格式 ((r,g,b),) 或 [[r,g,b]]
    :param location: 光源xyz位置，格式 ((x,y,z),) 或 [[x,y,z]]
    :return: 配置好的PointLights对象
    """
    # --------------------------------
    # 3 光照（使用官方默认参数，可通过传参覆盖）
    # --------------------------------
    lights = PointLights(
        device=device,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
        location=location
    )
    return lights

def rasterizer_set():
    # --------------------------------
    # 4 Rasterizer
    # --------------------------------
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    return raster_settings

def update_meshes_texture(
    original_meshes_list,
    tex,  # [1, C, H, W]
    target_index_list,
    device
):
    """
    用给定 texture 替换指定子 Mesh 的纹理

    Args:
        original_meshes_list: List[Meshes]
        tex: [1, C, H, W]
        target_index_list: 需要替换纹理的 mesh index
        device: torch.device

    Returns:
        new_meshes_list: List[Meshes]
    """

    new_meshes_list = []

    # ================= preprocess texture =================
    tex = tex.to(device).float()

    # [1, C, H, W] -> [1, H, W, C]
    tex = tex.permute(0, 2, 3, 1).contiguous()

    # ================= loop meshes =================
    for i, mesh in enumerate(original_meshes_list):

        # --------- 不在 target -> 直接 copy ----------
        if i not in target_index_list:
            new_meshes_list.append(mesh)
            continue

        # =====================================================
        # 目标 mesh：替换 texture
        # =====================================================

        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]

        # ================= 情况1：UV texture =================
        if isinstance(mesh.textures, TexturesUV):

            # UV 信息必须保留
            faces_uvs = mesh.textures.faces_uvs_padded()
            verts_uvs = mesh.textures.verts_uvs_padded()

            new_tex = TexturesUV(
                maps=tex,
                faces_uvs=faces_uvs,
                verts_uvs=verts_uvs
            )

        # ================= 情况2：Vertex color fallback =================
        elif isinstance(mesh.textures, TexturesVertex):

            # 用 texture 平均值替代 vertex color
            avg_color = tex.mean(dim=(1, 2), keepdim=True)  # [1,1,1,C]
            avg_color = avg_color.squeeze(1).squeeze(1)     # [1,C]

            verts_features = avg_color.expand(len(verts), -1)

            new_tex = TexturesVertex(
                verts_features=verts_features.unsqueeze(0)
            )

        # ================= fallback（无纹理） =================
        else:
            # 强制创建 UV texture（最安全）
            faces_uvs = mesh.textures.faces_uvs_padded()
            verts_uvs = mesh.textures.verts_uvs_padded()

            new_tex = TexturesUV(
                maps=tex,
                faces_uvs=faces_uvs,
                verts_uvs=verts_uvs
            )

        # ================= rebuild mesh =================
        new_mesh = mesh.__class__(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=new_tex
        )

        new_meshes_list.append(new_mesh)

    return new_meshes_list

def update_meshes_texture_dict(
    original_meshes_list,
    target_index_dict,     # {material_name: tex}
    material_names_list,   # List[str]
    device
):
    """
    用不同 texture 替换指定材质的 mesh

    Args:
        original_meshes_list: List[Meshes]
        target_index_dict: {mat_name: tex}, tex=[C,H,W] or [1,C,H,W]
        material_names_list: 每个 mesh 对应的材质名
        device: torch.device

    Returns:
        new_meshes_list: List[Meshes]
    """

    new_meshes_list = []

    for i, mesh in enumerate(original_meshes_list):

        mat_name = material_names_list[i]

        # ================= 不需要替换 =================
        if mat_name not in target_index_dict:
            new_meshes_list.append(mesh)
            continue

        # ================= 取对应 texture =================
        tex = target_index_dict[mat_name].to(device).float()

        # 统一 shape: [1,C,H,W]
        if tex.dim() == 3:
            tex = tex.unsqueeze(0)

        # [1,C,H,W] → [1,H,W,C]
        tex = tex.permute(0, 2, 3, 1).contiguous()

        # ================= rebuild =================
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]

        # ================= 情况1：UV =================
        if isinstance(mesh.textures, TexturesUV):

            faces_uvs = mesh.textures.faces_uvs_padded()
            verts_uvs = mesh.textures.verts_uvs_padded()

            new_tex = TexturesUV(
                maps=tex,
                faces_uvs=faces_uvs,
                verts_uvs=verts_uvs
            )

        # ================= 情况2：Vertex =================
        elif isinstance(mesh.textures, TexturesVertex):

            avg_color = tex.mean(dim=(1, 2), keepdim=True)  # [1,1,1,C]
            avg_color = avg_color.squeeze(1).squeeze(1)     # [1,C]

            verts_features = avg_color.expand(len(verts), -1)

            new_tex = TexturesVertex(
                verts_features=verts_features.unsqueeze(0)
            )

        # ================= fallback =================
        else:
            # ⚠️ 注意：这里必须保证有UV，否则会炸
            if hasattr(mesh.textures, "faces_uvs_padded"):

                faces_uvs = mesh.textures.faces_uvs_padded()
                verts_uvs = mesh.textures.verts_uvs_padded()

                new_tex = TexturesUV(
                    maps=tex,
                    faces_uvs=faces_uvs,
                    verts_uvs=verts_uvs
                )
            else:
                # 最保守 fallback
                avg_color = tex.mean(dim=(1, 2), keepdim=True).squeeze(1).squeeze(1)
                verts_features = avg_color.expand(len(verts), -1)

                new_tex = TexturesVertex(
                    verts_features=verts_features.unsqueeze(0)
                )

        # ================= new mesh =================
        new_mesh = mesh.__class__(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=new_tex
        )

        new_meshes_list.append(new_mesh)

    return new_meshes_list

def render_process(
    cameras,
    raster_settings,
    lights,
    meshes_list,   # ✅ List[Meshes]
    device: torch.device
) -> torch.Tensor:
    """
    多材质 Mesh 渲染（正确遮挡合成）
    返回: (B, 3, H, W)
    """

    B = cameras.R.shape[0]
    H ,W= raster_settings.image_size

    # ================= renderer =================
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    # ================= 初始化输出 =================
    final_rgb = torch.zeros((B, H, W, 3), device=device)
    final_depth = torch.full((B, H, W), float("inf"), device=device)

    # ================= 逐 mesh 渲染 =================
    for mesh in meshes_list:
        mesh = mesh.extend(B)  
        fragments = renderer.rasterizer(mesh)

        # depth: (B, H, W, K)
        depth = fragments.zbuf[..., 0]

        # mask
        mask = depth > 0

        # shader 颜色
        images = renderer.shader(fragments, mesh)
        rgb = images[..., :3]

        # ================= z-buffer 合成 =================
        update_mask = (depth < final_depth) & mask

        final_depth[update_mask] = depth[update_mask]
        final_rgb[update_mask] = rgb[update_mask]

    # ================= 格式转换 =================
    final_rgb = final_rgb.permute(0, 3, 1, 2)
    final_rgb = torch.clamp(final_rgb, 0.0, 1.0)

    return final_rgb,final_depth

def compose_with_background(
    rgb,          # (B,3,H,W)
    depth,        # (B,H,W)
    background    # (B,3,H,W)
):
    """
    使用 depth 做前景mask融合
    """

    # ================= 前景mask =================
    # 有效像素：深度不是inf
    mask = torch.isfinite(depth)   # (B,H,W)

    # 扩展到3通道
    mask = mask.unsqueeze(1)       # (B,1,H,W)

    # ================= 融合 =================
    output = torch.where(mask, rgb, background)

    return output,mask

def load_parma_and_render_main(
    object_mesh,   #  List[Meshes]
    background,
    path_camera_pose,
    image_size,
    device,
    fov=110,
    blur_radius=0.0,
    faces_per_pixel=1
):
    """
    支持多材质 mesh 渲染
    """

    # ================= 相机 =================
    cameras = generate_camera_from_params(
        pose_paths=path_camera_pose,
        device=device,
        fov=fov,
        img_size=image_size
    )

    # ================= 光照 =================
    light = light_set_fixed(device)

    # ================= raster =================
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=0
    )

    # ================= 渲染 =================
    image_tensor,images_depth = render_process(
        cameras=cameras,
        raster_settings=raster_settings,
        lights=light,
        meshes_list=object_mesh,  # 
        device=device
    )

    rendered_image_tensor,mask=compose_with_background(image_tensor,images_depth,background)

    # ================= debug =================
    visualize_and_save_render(image_tensor,save_dir="./exp/debug_results/2")
    visualize_and_save_render(rendered_image_tensor, save_dir="./exp/debug_results/1")

    return rendered_image_tensor,mask

# 输入camer参数，得路径
def load_parma_and_render_main_v2(
    object_mesh,   #  List[Meshes]
    background,
    cam_relative_pos,
    cam_relative_rot,
    image_size,

    device,
    fov=110,
    blur_radius=0.0,
    
    faces_per_pixel=1
):
    """
    支持多材质 mesh 渲染
    """

    # ================= 相机 =================
    cameras = generate_camera_from_params_v2(
        cam_relative_pos=cam_relative_pos,
        cam_relative_rot=cam_relative_rot,
        device=device,
        fov=fov,
        img_size=image_size
    )

    # ================= 光照 =================
    light = light_set_fixed(device)

    # ================= raster =================
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=0
    )

    # ================= 渲染 =================
    image_tensor,images_depth = render_process(
        cameras=cameras,
        raster_settings=raster_settings,
        lights=light,
        meshes_list=object_mesh,  # 
        device=device
    )

    rendered_image_tensor,mask=compose_with_background(image_tensor,images_depth,background)

    # ================= debug =================
    visualize_and_save_render(image_tensor,save_dir="./exp/debug_results/2")
    visualize_and_save_render(rendered_image_tensor, save_dir="./exp/debug_results/1")

    return rendered_image_tensor,mask



def main_debug2():

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
    OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/old.obj'  # OBJ模型路径
    OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/byd_yangwang.obj'  # OBJ模型路径
    # 
    # OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/TeslaModel3_blue.obj'
    # 图像尺寸（从RGB图自动获取，也可手动指定）
    IMG_SIZE = ( 720,1280)  # ( height,width)，若需自动获取可参考下方注释代码

    # -------------------------- 2. 初始化设备与路径检查 --------------------------
    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存目录
    

    # -------------------------- 3. 加载模型与相机参数 --------------------------


    mesh_model = load_obj_model(OBJECT_OBJ_PATH, device)

    camera_paths_list=[POSE_PATH,POSE_PATH.replace("fixed_002","fixed_001")] 
    backgroud_paths_list=[RGB_PATH,RGB_PATH.replace("fixed_002","fixed_001")]

    backgroud_images=load_background_images(backgroud_paths_list,device=device)
    # 生成对应大小的render texture
    # 随机
    # 读取图像
    tex_images=cv2.imread("./debug_results/controlnet_sample.jpg")
    tex_images=cv2.resize(tex_images,(IMG_SIZE[1],IMG_SIZE[0]))
    tex_images=torch.from_numpy(tex_images).permute(2,0,1).float().to(device)
    # 归一化
    tex=tex_images/255.0
    tex=tex.unsqueeze(0)

    # tex=torch.ones_like(backgroud_images).to(device)

    # new_mesh = mesh_model.extend(len(camera_paths_list))

    images_rnedered = load_parma_and_render_main(object_mesh=mesh_model,
                                                 background=backgroud_images,
                                                 path_camera_pose=camera_paths_list,
                                                 image_size=IMG_SIZE,
                                                 device=device,
                                                 fov=110,
                                                 blur_radius=0.0,
                                                 faces_per_pixel=1)
    
    new_mesh1=update_meshes_texture(
        original_meshes_list=mesh_model,
        tex=tex,           # 形状为 [1, C, H, W] 的纹理张量
        target_index_list=[1,3],
        device=device)
    
    images_rnedered = load_parma_and_render_main(object_mesh=new_mesh1,
                                                 background=backgroud_images,
                                                 path_camera_pose=camera_paths_list,
                                                 image_size=IMG_SIZE,
                                                 device=device,
                                                 fov=110,
                                                 blur_radius=0.0,
                                                 faces_per_pixel=1)
    
    #
    


def main_debug3():


    # -------------------------- 1. 配置路径与参数 --------------------------
    # 数据路径（按你的需求指定）
    root="/root/autodl-fs/data_debug/carla_sample_data20260411_final/vehicle_audi_tt/location_000/"
    name="fixed_006"
    RGB_PATH = os.path.join(root, "rgb", name+".png")       # RGB图路径,rgb
    DEPTH_PATH = os.path.join(root, "depth", name+".png")   # 深度图路径
    MASK_PATH = os.path.join(root, "mask", name+".png")     # 掩码图路径  
    POSE_PATH = os.path.join(root, "camera_pose", name+".npz")   # 位姿文件路径
    INTRINSICS_PATH = os.path.join(root, "camera_intrinsics", name+".npz")  # 内参文件路径
    SAVE_DIR = "./debug_results/exp2"  # 结果保存目录
    OBJECT_OBJ_PATH = '/root/autodl-fs/data_debug/rect/Untitled.obj'  # OBJ模型路径
    # OBJECT_OBJ_PATH = '/root/autodl-fs/MM3DAdv_data/3Dmodel/mazda/mazda_part.obj'  # OBJ模型路径
    # 
    # OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/TeslaModel3_blue.obj'
    # 图像尺寸（从RGB图自动获取，也可手动指定）
    IMG_SIZE = ( 720,1280)  # ( height,width)，若需自动获取可参考下方注释代码

    # -------------------------- 2. 初始化设备与路径检查 --------------------------
    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存目录
    

    # -------------------------- 3. 加载模型与相机参数 --------------------------


    mesh_model ,material_list= load_obj_model_return_mesh_material(OBJECT_OBJ_PATH, device)

    camera_paths_list=[POSE_PATH,POSE_PATH.replace("fixed_002","fixed_001")] 
    backgroud_paths_list=[RGB_PATH,RGB_PATH.replace("fixed_002","fixed_001")]

    backgroud_images=load_background_images(backgroud_paths_list,device=device)
    # 生成对应大小的render texture
    # 随机
    # 读取图像
    tex_images=cv2.imread("./test_imgs/dog2.png")
    tex_images=cv2.cvtColor(tex_images,cv2.COLOR_BGR2RGB)
    tex_images=cv2.resize(tex_images,(IMG_SIZE[1],IMG_SIZE[0]))
    tex_images=torch.from_numpy(tex_images).permute(2,0,1).float().to(device)
    # 归一化
    tex=tex_images/255.0
    tex=tex.unsqueeze(0)

    # tex=torch.ones_like(backgroud_images).to(device)

    # new_mesh = mesh_model.extend(len(camera_paths_list))

    images_rnedered = load_parma_and_render_main(object_mesh=mesh_model,
                                                 background=backgroud_images,
                                                 path_camera_pose=camera_paths_list,
                                                 image_size=IMG_SIZE,
                                                 device=device,
                                                 fov=110,
                                                 blur_radius=0.0,
                                                 faces_per_pixel=1)
    target_index_dict={}
    target_list=[1,3]
    for i,material in enumerate(material_list):
        target_index_dict[material]=tex

    new_mesh1=update_meshes_texture_dict(
        original_meshes_list=mesh_model,
        target_index_dict=target_index_dict,           # 形状为 [1, C, H, W] 的纹理张量
        material_names_list=material_list,
        device=device)
    

    images_rnedered = load_parma_and_render_main(object_mesh=new_mesh1,
                                                 background=backgroud_images,
                                                 path_camera_pose=camera_paths_list,
                                                 image_size=IMG_SIZE,
                                                 device=device,
                                                 fov=110,
                                                 blur_radius=0.0,
                                                 faces_per_pixel=1)
    
    #
  

if __name__ == "__main__":
    main_debug3()
