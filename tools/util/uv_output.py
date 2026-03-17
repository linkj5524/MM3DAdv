import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def export_uv_layout(
    obj_path: str,
    output_size: tuple = (1024, 1024),
    output_path: str = "uv_layout.png",
    show_texture: bool = False,
    texture_path: str = None
):
    """
    读取 OBJ 模型（兼容单网格/多网格），导出 UV 展开图
    :param obj_path: OBJ 模型路径
    :param output_size: UV 图输出尺寸 (宽, 高)
    :param output_path: 输出图像路径
    :param show_texture: 是否叠加纹理图像（需指定 texture_path）
    :param texture_path: 纹理图像路径（与 UV 绑定的纹理）
    """
    # 1. 读取 OBJ 模型（强制返回单个网格对象）
    mesh = trimesh.load(obj_path, process=False)
    
    # 关键修复：将 Scene 转换为单个合并的 Trimesh 对象
    if isinstance(mesh, trimesh.Scene):
        # 合并场景中的所有网格（保留 UV 坐标）
        merged_meshes = []
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and hasattr(geom.visual, 'uv'):
                merged_meshes.append(geom)
        if not merged_meshes:
            raise ValueError("Scene 中未找到包含 UV 坐标的网格！")
        # 合并多个网格为一个
        mesh = trimesh.util.concatenate(merged_meshes)
    
    # 检查是否为有效网格且包含 UV 坐标
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("OBJ 模型加载后不是有效网格！")
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        raise ValueError("OBJ 模型未包含 UV 坐标！")
    
    # 2. 提取 UV 坐标和面信息
    uv_coords = mesh.visual.uv  # [N, 2] 所有顶点的 UV 坐标（U, V）
    faces = mesh.faces          # [M, 3] 面的顶点索引
    uv_faces = uv_coords[faces] # [M, 3, 2] 每个面的 3 个顶点 UV 坐标
    
    # 3. UV 坐标归一化（确保在 0~1 范围内）
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_coords_normalized = (uv_coords - uv_min) / (uv_max - uv_min + 1e-8)  # 避免除零
    
    # 4. 创建 UV 展开图画布
    uv_img = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255  # 白色背景
    if show_texture and texture_path:
        # 加载纹理图像并缩放至输出尺寸
        try:
            texture = Image.open(texture_path).resize(output_size)
            uv_img = np.array(texture)
        except Exception as e:
            print(f"加载纹理失败：{e}，使用白色背景")
    
    # 5. 将 UV 网格绘制到画布（转换 UV → 像素坐标）
    # UV (0,0) → 画布左上，UV (1,1) → 画布右下（注意 V 轴翻转，因图像 V 轴向下）
    def uv_to_pixel(uv):
        x = int(uv[0] * (output_size[0] - 1))
        y = int((1 - uv[1]) * (output_size[1] - 1))  # 翻转 V 轴
        return (x, y)
    
    # 绘制每个面的 UV 三角网格（红色线条）
    plt.figure(figsize=(output_size[0]/100, output_size[1]/100), dpi=100)
    plt.imshow(uv_img)
    
    for face_uv in uv_faces:
        # 归一化当前面的 UV 坐标
        face_uv_normalized = (face_uv - uv_min) / (uv_max - uv_min + 1e-8)
        # 转换为像素坐标
        p1 = uv_to_pixel(face_uv_normalized[0])
        p2 = uv_to_pixel(face_uv_normalized[1])
        p3 = uv_to_pixel(face_uv_normalized[2])
        # 绘制三角形边
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=0.5)
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color='red', linewidth=0.5)
        plt.plot([p3[0], p1[0]], [p3[1], p1[1]], color='red', linewidth=0.5)
    
    # 6. 配置绘图并保存
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    print(f"UV 展开图已导出至：{output_path}")
    return uv_coords_normalized, uv_img

# ====================== 调用示例 ======================
if __name__ == "__main__":
    # 替换为你的 OBJ 模型路径
    obj_path = "/root/autodl-fs/data/object_model/TeslaModel3_blue.obj"

    
    # 导出纯 UV 网格图
    export_uv_layout(
        obj_path=obj_path,
        output_size=(1024, 1024),
        output_path="./debug_results/uv_layout.png",
        show_texture=False
    )
    
