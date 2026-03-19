import torch
import os
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

# ====================== 1. 核心函数定义 ======================
def apply_texture_to_mesh(
    object_mesh,
    texture,  # 纹理张量 (B, C, H, W) 或 None，C=3，范围[0,1]
    device):
    """
    将纹理张量（B×C×H×W）映射到 3D 网格（Meshes），返回带纹理的新 Meshes 对象
    核心逻辑：统一使用顶点颜色映射（无需UV）
    """
    # 1. 边界条件：无纹理时返回原 mesh 副本
    if texture is None:
        return object_mesh.clone()

    # 2. 复制原始 mesh，避免修改原对象
    new_mesh = object_mesh.clone().to(device)
    
    # 3. 纹理格式转换：B×C×H×W → B×H×W×C（适配 PyTorch3D 纹理格式）
    texture = texture.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 3)
    texture = texture.to(device)
    texture = texture.to(dtype=torch.float32) 
    
    # ========== 顶点颜色映射（无需UV） ==========
    # 获取每个 mesh 的顶点数
    verts_padded = new_mesh.verts_padded()  # (B, V, 3)
    B, V, _ = verts_padded.shape
    
    # 将纹理图像平均采样到顶点（适配批量）
    vertex_colors = []
    for b in range(B):
        # 单批次纹理：(H, W, 3) → 展平为 (H*W, 3)
        tex_flat = texture[b].reshape(-1, 3)
        # 均匀采样到顶点（避免随机采样的不确定性）
        sample_step = max(1, tex_flat.shape[0] // V)
        sample_idx = torch.arange(0, tex_flat.shape[0], sample_step)[:V].to(device)
        vert_color = tex_flat[sample_idx]  # (V, 3)
        # 补充不足的顶点（若纹理像素数 < 顶点数）
        if len(vert_color) < V:
            pad_num = V - len(vert_color)
            vert_color = torch.cat([vert_color, vert_color[:pad_num]], dim=0)
        vertex_colors.append(vert_color)
    
    # 拼接为批量顶点颜色：(B, V, 3)
    vertex_colors = torch.stack(vertex_colors, dim=0).to(device)
    # 创建顶点颜色纹理
    texture_vertex = TexturesVertex(verts_features=vertex_colors)
    new_mesh.textures = texture_vertex

    return new_mesh


def extract_texture_from_mesh_no_uv(
    textured_mesh: Meshes,
    texture_size: tuple = (512, 512),  # 输出纹理图像尺寸 (W, H)
    device=None
) -> torch.Tensor:
    """
    无需UV坐标，直接从顶点颜色逆向生成纹理图像（基于顶点空间分布，无高斯模糊）
    """
    if device is None:
        device = textured_mesh.device
    
    B = len(textured_mesh)  # 批量大小
    W, H = texture_size
    texture_images = []

    # 遍历每个批量的 mesh
    for b in range(B):
        single_mesh = textured_mesh[b:b+1]  # 单批次 mesh
        textures = single_mesh.textures

        # 提取顶点颜色（优先用纹理顶点颜色，无则用顶点坐标归一化）
        if isinstance(textures, TexturesVertex):
            vert_colors = textures.verts_features_padded()[0]  # (V, 3) 纹理顶点颜色
        else:
            # 用顶点坐标归一化作为颜色（兜底方案）
            verts = single_mesh.verts_padded()[0]  # (V, 3) 顶点坐标
            vert_colors = (verts - verts.min()) / (verts.max() - verts.min() + 1e-8)
        
        # 顶点坐标归一化到 [0,1]（用于映射到纹理画布）
        verts = single_mesh.verts_padded()[0]  # (V, 3)
        verts_normalized = (verts - verts.min()) / (verts.max() - verts.min() + 1e-8)
        
        # 创建空的纹理画布（H, W, 3）
        tex_canvas = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        count_canvas = torch.zeros((H, W), dtype=torch.float32, device=device) + 1e-8

        # 核心：将顶点颜色映射到纹理画布（基于顶点的 X/Y 坐标）
        # X → 纹理宽度（U），Y → 纹理高度（V）
        for i in range(len(vert_colors)):
            # 顶点归一化坐标 → 像素坐标
            x_norm = verts_normalized[i, 0]  # X轴归一化值 [0,1]
            y_norm = verts_normalized[i, 1]  # Y轴归一化值 [0,1]
            
            # 转换为像素坐标
            u = int(x_norm * (W - 1))
            v = int((1 - y_norm) * (H - 1))  # 翻转Y轴（图像向下）
            
            # 边界检查
            u = max(0, min(W-1, u))
            v = max(0, min(H-1, v))
            
            # 填充颜色
            tex_canvas[v, u] += vert_colors[i]
            count_canvas[v, u] += 1

        # 归一化重叠顶点的颜色（平均）
        tex_canvas = tex_canvas / count_canvas.unsqueeze(-1)
        
        # 转换为 (3, H, W) 格式，限制范围 [0,1]
        tex_img = tex_canvas.permute(2, 0, 1).clamp(0.0, 1.0)
        texture_images.append(tex_img)
    
    # 拼接批量纹理图像：(B, 3, H, W)
    texture_batch = torch.stack(texture_images, dim=0)
    return texture_batch


def load_obj_mesh_no_uv(obj_path, device):
    """
    加载 OBJ 模型（无需UV坐标），返回 Meshes 对象
    """
    # 加载 OBJ 文件（不依赖纹理/UV，仅加载几何信息）
    verts, faces, aux = load_obj(
        obj_path,
        load_textures=False,  # 关闭纹理加载（避免UV依赖）
        device=device
    )
    
    # 创建基础 Meshes 对象（无纹理）
    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx]
    )
    
    return mesh.to(device)


def save_texture_image(texture_tensor, save_path):
    """
    将纹理张量 (3, H, W) 保存为 PNG 图像
    """
    # 转换为 (H, W, 3) 并归一化到 [0, 255]
    texture_np = texture_tensor.permute(1, 2, 0).cpu().numpy()
    texture_np = (texture_np * 255).astype('uint8')
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图像
    plt.imsave(save_path, texture_np)
    print(f"纹理图像已保存至：{save_path}")

# ====================== 2. 主流程执行 ======================
if __name__ == "__main__":
    # 配置参数
    obj_path = "/root/autodl-fs/data/object_model/byd_yangwang.obj"  # 你的 OBJ 路径
    texture_size = (512, 512)  # 输出纹理尺寸
    save_path = "./debug_results/extracted_texture_no_uv.png"  # 保存路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 步骤1：加载 OBJ 模型（无需UV）
    print("正在加载 OBJ 模型（无需UV）...")
    mesh = load_obj_mesh_no_uv(obj_path, device)
    print(f"模型加载完成：顶点数={len(mesh.verts_padded()[0])}，面数={len(mesh.faces_padded()[0])}")
    
    # 步骤2：给模型绑定自定义纹理（可选，演示用）
    # 生成随机测试纹理 (B=1, 3, H, W)
    test_texture = torch.rand((1, 3, texture_size[1], texture_size[0]), device=device)
    textured_mesh = apply_texture_to_mesh(mesh, test_texture, device)
    print("自定义纹理绑定完成")
    
    # 步骤3：逆向提取纹理图像（无需UV，无高斯模糊）
    print("正在逆向提取纹理图像...")
    extracted_texture = extract_texture_from_mesh_no_uv(
        textured_mesh=textured_mesh,
        texture_size=texture_size,
        device=device
    )
    
    # 步骤4：保存纹理图像
    save_texture_image(extracted_texture[0], save_path)
    



# import torch
# import os
# from pytorch3d.io import load_obj
# from pytorch3d.structures import Meshes

# def parse_obj_components_from_text(obj_path):
#     """
#     手动解析OBJ文本，精准提取面信息和组件映射（修复面数量不匹配问题）
#     返回：
#         component_names: 组件名称列表
#         face_component_idx: 每个有效面对应的组件索引（tensor）
#         valid_face_lines: 有效面行的索引（用于校验）
#     """
#     component_names = []
#     current_component = "default"
#     face_component_map = []  # 仅记录有效面
#     valid_face_count = 0     # 有效面数量（排除注释/空行）
    
#     # 读取OBJ文本并记录每行类型
#     with open(obj_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     # 第一步：先统计所有组件名称
#     for line in lines:
#         line = line.strip()
#         if not line or line.startswith('#'):
#             continue
#         # 识别组件标签（o 或 g）
#         if line.startswith('o '):
#             comp_name = line.split(' ', 1)[1].strip()
#             if comp_name not in component_names:
#                 component_names.append(comp_name)
#         elif line.startswith('g '):
#             comp_name = line.split(' ', 1)[1].strip()
#             if comp_name not in component_names:
#                 component_names.append(comp_name)
    
#     # 若无组件标签，默认添加default
#     if not component_names:
#         component_names = ["default"]
#         current_component = "default"
#     else:
#         current_component = component_names[0]  # 默认第一个组件

#     # 第二步：重新遍历，只记录有效面的组件映射
#     for line in lines:
#         line = line.strip()
#         if not line or line.startswith('#'):
#             continue
        
#         # 更新当前组件
#         if line.startswith('o '):
#             current_component = line.split(' ', 1)[1].strip()
#             if current_component not in component_names:
#                 component_names.append(current_component)
#         elif line.startswith('g '):
#             current_component = line.split(' ', 1)[1].strip()
#             if current_component not in component_names:
#                 component_names.append(current_component)
        
#         # 仅处理有效面（f 开头，且有顶点索引）
#         elif line.startswith('f '):
#             # 过滤空面（理论上不会出现，但防御性处理）
#             if len(line.split()) < 4:
#                 continue
#             valid_face_count += 1
#             # 获取当前组件索引
#             comp_idx = component_names.index(current_component)
#             face_component_map.append(comp_idx)

#     # 转换为tensor（确保类型匹配）
#     face_component_idx = torch.tensor(face_component_map, dtype=torch.long)
#     return component_names, face_component_idx, valid_face_count

# def analyze_obj_components(obj_path, device=None):
#     """
#     加载OBJ文件，分析并打印所有组件信息（彻底修复索引不匹配问题）
#     """
#     # 1. 设备初始化
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"===== OBJ 文件分析开始 =====")
#     print(f"文件路径: {obj_path}")
#     print(f"使用设备: {device}")
#     print("-" * 50)

#     # 2. 基础加载（无split参数）
#     try:
#         verts, faces, aux = load_obj(
#             obj_path,
#             load_textures=False,
#             device=device
#         )
#     except Exception as e:
#         print(f"❌ 加载OBJ失败: {e}")
#         return {}

#     # 3. 手动解析组件信息（核心修复）
#     component_names, face_component_idx, parsed_face_count = parse_obj_components_from_text(obj_path)
    
#     # 4. 关键：对齐面数量（修复索引不匹配）
#     loaded_face_count = len(faces.verts_idx) if faces.verts_idx is not None else 0
#     print(f"🔍 面数量校验:")
#     print(f"  - 解析OBJ文本得到有效面数: {parsed_face_count}")
#     print(f"  - load_obj加载得到面数: {loaded_face_count}")
    
#     # 取最小面数，截断过长的索引（兼容不规则OBJ）
#     min_face_count = min(parsed_face_count, loaded_face_count)
#     if min_face_count != parsed_face_count or min_face_count != loaded_face_count:
#         print(f"⚠️  面数量不匹配，自动截断至最小面数: {min_face_count}")
#         face_component_idx = face_component_idx[:min_face_count]
#         faces_verts_idx = faces.verts_idx[:min_face_count]  # 截断加载的面
#     else:
#         faces_verts_idx = faces.verts_idx
#         print(f"✅ 面数量校验通过")

#     # 5. 基础信息统计
#     total_verts = len(verts)
#     total_faces = min_face_count
#     print("-" * 50)
#     print(f"📊 模型基础信息:")
#     print(f"  - 总顶点数: {total_verts:,}")
#     print(f"  - 有效面数: {total_faces:,}")
#     print("-" * 50)

#     # 6. 组件检测结果
#     is_multi_component = len(component_names) > 1
#     print(f"🔍 组件检测结果:")
#     print(f"  - 是否多组件: {'✅ 是' if is_multi_component else '❌ 否'}")
#     print(f"  - 组件总数: {len(component_names)}")
#     print("-" * 50)

#     # 7. 逐个分析组件（修复索引逻辑）
#     components = {}
#     print(f"📋 各组件详细信息:")
#     for idx, name in enumerate(component_names):
#         # 筛选当前组件的面（使用截断后的索引）
#         if total_faces == 0:
#             print(f"  - [{idx+1}] {name}: 空组件（无面）→ 跳过")
#             continue
        
#         # 生成掩码（确保掩码长度匹配）
#         face_mask = (face_component_idx == idx)
#         if len(face_mask) != len(faces_verts_idx):
#             face_mask = face_mask[:len(faces_verts_idx)]
        
#         # 筛选面
#         component_faces = faces_verts_idx[face_mask]
#         component_face_num = len(component_faces)

#         # 跳过空组件
#         if component_face_num == 0:
#             print(f"  - [{idx+1}] {name}: 空组件（无面）→ 跳过")
#             continue

#         # 构建组件Meshes对象
#         component_mesh = Meshes(
#             verts=[verts],  # 共享顶点池
#             faces=[component_faces]
#         ).to(device)

#         # 统计实际使用的顶点数
#         component_vert_used = len(torch.unique(component_faces)) if component_face_num > 0 else 0
#         components[name] = component_mesh

#         # 打印详情
#         print(f"  - [{idx+1}] 组件名称: {name}")
#         print(f"    · 实际使用顶点数: {component_vert_used:,}")
#         print(f"    · 面数: {component_face_num:,}")
#         print(f"    · 占总面数比例: {component_face_num/total_faces*100:.2f}%")

#     print("-" * 50)
#     print(f"===== OBJ 文件分析完成 =====")
#     return components

# # ====================== 主程序执行 ======================
# if __name__ == "__main__":
#     # 配置OBJ路径
#     OBJECT_OBJ_PATH = '/root/autodl-fs/data/object_model/byd_yangwang.obj'
    
#     # 执行分析
#     if not os.path.exists(OBJECT_OBJ_PATH):
#         print(f"❌ OBJ文件不存在: {OBJECT_OBJ_PATH}")
#     else:
#         components = analyze_obj_components(OBJECT_OBJ_PATH)