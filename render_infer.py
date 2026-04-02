import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader


from adv_attack.pt3D_tools import *
from adv_attack.util import *


def count_false_positive_single_model(
    model_result: dict,
    ref_result: dict,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5
) -> dict:
    """
    单个模型版本：计算每个batch的误识别框数量 & 平均误识别概率
    输入格式与你原来的 count_matched_results 完全相同
    误识别定义：置信度达标，但 类别错误 或 IoU不达标 或 参考无框却检出框

    返回：
        {
            "false_pos_counts": [0, 2, 1, 0, ...],  # 每个batch误识别框数量
            "avg_false_pos_prob": 0.xx              # 平均每个batch误识别概率
        }
    """
    required_keys = ['boxes', 'scores', 'labels']
    for key in required_keys:
        if key not in ref_result or key not in model_result:
            raise ValueError(f"缺少关键字段：{key}")

    ref_batch_num = len(ref_result['labels'])
    model_batch_num = len(model_result['labels'])

    if ref_batch_num != model_batch_num:
        return {
            "false_pos_counts": [0] * ref_batch_num,
            "avg_false_pos_prob": 0.0
        }

    # 提取参考batch的目标框
    ref_batch_features = []
    for batch_idx in range(ref_batch_num):
        if len(ref_result['labels'][batch_idx]) == 0 or len(ref_result['boxes'][batch_idx]) == 0:
            ref_batch_features.append(None)
            continue

        ref_box = ref_result['boxes'][batch_idx][0]
        ref_label = ref_result['labels'][batch_idx][0]
        ref_batch_features.append({
            'box': ref_box,
            'label': ref_label
        })

    false_pos_counts = []

    # 遍历每个batch，计算误识别
    for batch_idx in range(ref_batch_num):
        ref_feat = ref_batch_features[batch_idx]

        curr_labels = model_result['labels'][batch_idx]
        curr_boxes = model_result['boxes'][batch_idx]
        curr_scores = model_result['scores'][batch_idx]

        false_pos = 0

        for box_idx in range(len(curr_labels)):
            c_score = curr_scores[box_idx]
            if c_score < conf_threshold:
                continue  # 只统计置信度达标的框

            c_label = curr_labels[box_idx]
            c_box = curr_boxes[box_idx]

            # ====== 误识别规则 ======
            if ref_feat is None:
                # 参考无框 → 检出就算误识别
                false_pos += 1
            else:
                # 类别不匹配 或 IoU不匹配 → 误识别
                label_match = torch.equal(c_label, ref_feat['label']) if torch.is_tensor(c_label) else (c_label == ref_feat['label'])
                iou_match = calculate_box_iou(c_box, ref_feat['box']) >= iou_threshold

                if not label_match or not iou_match:
                    false_pos += 1

        false_pos_counts.append(false_pos)

    # 平均误识别概率 = 总误识别数 / 总batch数
    total_batches = len(false_pos_counts)
    total_false_pos = sum(false_pos_counts)
    avg_false_pos_prob = round(total_false_pos / total_batches, 4) if total_batches > 0 else 0.0

    return {
        "false_pos_counts": false_pos_counts,
        "avg_false_pos_prob": avg_false_pos_prob
    }



    

# ==================================================
# 【推理配置】
# ==================================================
CONFIG_PATH = "configs/render_infer.yaml"    # 你的实验 yaml


# ==================================================
# 加载配置
# ==================================================
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

ADV_TEXTURE_PATH = cfg["texture_path"]
SAVE_ROOT =  cfg["save_path"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ======================
# 1. 加载数据集
# ======================
train_loader, _ = build_RGBCameraPose_dataloader(
    train_root_dir=cfg["train_root_dir"],
    val_root_dir=cfg["val_root_dir"],
    batch_size=cfg["batch_size"],
    num_workers=cfg["num_workers"],
    transform=None
)


# ======================
# 2. 加载 3D 模型
# ======================
mesh_model, material_list = load_obj_model_return_mesh_material(
    cfg["mesh_model_path"], device
)
target_materials = cfg["target_material_dict"]

# ======================
# 3. 加载优化好的纹理
# ======================
adv_texture = torch.load(ADV_TEXTURE_PATH, map_location=device)
print(f" Loaded texture: {adv_texture.shape}")

# ======================
# 4. 开始批量渲染
# ======================
os.makedirs(SAVE_ROOT, exist_ok=True)
render_size = (cfg["render_size"]["height"], cfg["render_size"]["width"])
detect_size = cfg["image_size"]

print("\n Start rendering all images ...")

with torch.no_grad():
    for batch_idx, (bg_images, cam_paths) in enumerate(tqdm(train_loader)):
        bg_images = bg_images.to(device)

        for i in range(len(bg_images)):
            bg = bg_images[i:i+1]
            cam_path = [cam_paths[i]]

            # --------------------------
            # A. 原始纹理渲染
            # --------------------------
            ori_render = load_parma_and_render_main(
                object_mesh=mesh_model,
                background=bg,
                path_camera_pose=cam_path,
                image_size=render_size,
                device=device,
                fov=110
            )
            ori_render = resize_tensor_ratio_pad(ori_render, detect_size, detect_size)

            # --------------------------
            # B. 对抗纹理渲染
            # --------------------------
            # for index_adv in range(len(target_materials)):
            #         tex_dict={target_materials[index_adv]: adv_texture[index_adv]}
            

            if len(target_materials) == len(adv_texture):
                # 长度相等：一一对应赋值
                tex_dict = {}
                for mat, tex in zip(target_materials, adv_texture):
                    tex_dict[mat] = tex
            else:
                # 长度不等：所有材质都使用同一张纹理（取第 0 张）
                adv_tex = adv_texture[0] if adv_texture.ndim == 4 else adv_texture
                tex_dict = {mat: adv_tex for mat in target_materials}



            adv_mesh = update_meshes_texture_dict(
                original_meshes_list=mesh_model,
                target_index_dict=tex_dict,
                material_names_list=material_list,
                device=device
            )
            adv_render = load_parma_and_render_main(
                object_mesh=adv_mesh,
                background=bg,
                path_camera_pose=cam_path,
                image_size=render_size,
                device=device,
                fov=110
            )



            # --------------------------
            # 保存
            # --------------------------
            save_dir = f"{SAVE_ROOT}/img_{batch_idx}_{i}"
            os.makedirs(save_dir, exist_ok=True)

            save_image(ori_render.clamp(0,1), f"{save_dir}/original.png")
            save_image(adv_render.clamp(0,1), f"{save_dir}/adv_render.png")

print(f"\n All done! Results saved to: {SAVE_ROOT}")