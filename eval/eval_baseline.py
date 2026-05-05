import os
import re
import yaml
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, maskrcnn_resnet50_fpn
from ultralytics import YOLO

# ===================== 全局配置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IOU_THRESH = 0.5
CONF_THRESH = 0.5

# 类别ID对齐
YOLO_CAR_ID = 2
TORCHVISION_CAR_ID = 3

# ===================== 加载 4 大模型 =====================
def load_all_models():
    models = {}
    models["YOLO-V5"] = YOLO("yolov5s.pt").to(DEVICE)
    faster = fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE)
    faster.eval()
    models["Faster RCNN"] = faster
    ssd = ssd300_vgg16(pretrained=True).to(DEVICE)
    ssd.eval()
    models["SSD"] = ssd
    maskrcnn = maskrcnn_resnet50_fpn(pretrained=True).to(DEVICE)
    maskrcnn.eval()
    models["Mask RCNN"] = maskrcnn
    return models

# ===================== 仅推理，返回过滤后车辆像素框 =====================
def infer_get_car_boxes(model_name, model, img_path):
    img = Image.open(img_path).convert("RGB")
    boxes = []

    if model_name == "YOLO-V5":
        res = model(img, conf=CONF_THRESH)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            boxes.append((x1, y1, x2, y2, conf, cls))
    else:
        transform = torchvision.transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(img_tensor)[0]
        for idx, box in enumerate(pred["boxes"]):
            cls = int(pred["labels"][idx].item())
            conf = pred["scores"][idx].item()
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = box.tolist()
            boxes.append((x1, y1, x2, y2, conf, cls))

    # 过滤车辆并统一ID为2
    car_boxes = []
    for b in boxes:
        cls = b[5]
        if (model_name == "YOLO-V5" and cls == YOLO_CAR_ID) or \
           (model_name != "YOLO-V5" and cls == TORCHVISION_CAR_ID):
            car_boxes.append((b[0], b[1], b[2], b[3], b[4], YOLO_CAR_ID))
    return car_boxes

# ===================== IoU =====================
def calc_iou(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x1g, y1g, x2g, y2g = box2[:4]
    xx1 = max(x1, x1g)
    yy1 = max(y1, y1g)
    xx2 = min(x2, x2g)
    yy2 = min(y2, y2g)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter
    return inter / union if union > 1e-8 else 0.0

# ===================== 单样本评估：扣除原图固有FP =====================
def evaluate_one_sample(gt_norm, ori_boxes, adv_boxes, img_w, img_h):
    # GT 归一化转像素
    x1g, y1g, x2g, y2g = gt_norm
    gt_box = (x1g * img_w, y1g * img_h, x2g * img_w, y2g * img_h)

    # -------- 1. 计算原图固有假阳性 FP_ori --------
    fp_ori = 0
    ori_has_match = False
    for p in ori_boxes:
        iou = calc_iou(p, gt_box)
        if iou >= IOU_THRESH and not ori_has_match:
            ori_has_match = True
        else:
            fp_ori += 1

    # -------- 2. 计算对抗图原始假阳性 FP_adv_raw、TP、FN --------
    tp = 0
    fp_adv_raw = 0
    adv_has_match = False
    for p in adv_boxes:
        iou = calc_iou(p, gt_box)
        if iou >= IOU_THRESH and not adv_has_match:
            tp = 1
            adv_has_match = True
        else:
            fp_adv_raw += 1

    fn = 1 if not adv_has_match else 0

    # -------- 3. 关键：扣除原图自带FP，得到有效增量FP --------
    fp_effective = max(0, fp_adv_raw - fp_ori)

    precision = tp / (tp + fp_effective) if (tp + fp_effective) > 0 else 0.0
    return precision, tp, fp_effective, fn

# ===================== 主评估流程 =====================
def run_eval(root_dir):
    all_models = load_all_models()
    model_names = list(all_models.keys())
    pattern = re.compile(r"detect_b(\d+)_(\d+)")

    dirs = [d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and pattern.match(d)]
    total_samples = len(dirs)
    print(f"总样本数: {total_samples}")

    res_summary = {
        name: {"P@0.5":0.0, "TP":0, "FP":0, "FN":0, "ASR":0.0}
        for name in model_names
    }

    for dname in dirs:
        sample_dir = os.path.join(root_dir, dname)
        ori_img_path = os.path.join(sample_dir, "origin.jpg")
        adv_img_path = os.path.join(sample_dir, "adv.jpg")
        gt_txt_path = os.path.join(sample_dir, "gt.txt")

        if not all(os.path.exists(p) for p in [ori_img_path, adv_img_path, gt_txt_path]):
            continue

        # 读GT
        with open(gt_txt_path, "r") as f:
            gt_norm = list(map(float, f.readline().strip().split()))
        if len(gt_norm) < 4:
            continue

        w, h = Image.open(ori_img_path).size

        for m_name in model_names:
            model = all_models[m_name]
            # 同时推理原图、对抗图
            ori_boxes = infer_get_car_boxes(m_name, model, ori_img_path)
            adv_boxes = infer_get_car_boxes(m_name, model, adv_img_path)

            p, tp, fp, fn = evaluate_one_sample(gt_norm, ori_boxes, adv_boxes, w, h)

            res_summary[m_name]["TP"] += tp
            res_summary[m_name]["FP"] += fp
            res_summary[m_name]["FN"] += fn

    # 汇总计算最终指标
    print("\n===== 修正版评估（FP已扣除原图固有误检）=====")
    for m in model_names:
        tp = res_summary[m]["TP"]
        fp = res_summary[m]["FP"]
        fn = res_summary[m]["FN"]

        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        asr = fn / total_samples if total_samples > 0 else 0.0

        res_summary[m]["P@0.5"] = round(p_val, 4)
        res_summary[m]["ASR"] = round(asr, 4)

        print(f"{m:15s} | P@0.5:{p_val:.4f} ASR:{asr:.4f} TP:{tp} FP:{fp} FN:{fn}")

    # 保存YAML
    yaml_path = "/root/autodl-fs/MM3Dadv_exp/val/attack_val_20260502/baseline/P05_results.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(res_summary, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"\n结果已保存: {yaml_path}")

    return res_summary

# ===================== 运行 =====================
if __name__ == "__main__":
    ROOT = "/root/autodl-fs/MM3Dadv_exp/val/attack_val_20260502/baseline/visual_val"
    run_eval(ROOT)