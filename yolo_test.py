# import cv2
# import torch
# from ultralytics import YOLO
# import numpy as np

# def main(image_path, model_path="yolov11n.pt"):
#     # 1. 加载YOLOv11模型
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     print(f"使用设备: {device}")

#     # 2. 使用OpenCV读取图像
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图像: {image_path}")
    
#     # 3. 图像预处理
#     # 3.1 转换颜色空间 (BGR -> RGB)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_rgb = cv2.resize(img_rgb, (640, 640))
#     # 3.2 转换为Tensor并调整维度
#     # 转换为float32类型并归一化到[0, 1]
#     tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
#     # 添加批次维度 [C, H, W] -> [1, C, H, W]
#     tensor = tensor.unsqueeze(0).to(device)

#     # 4. 执行目标检测
#     results = model(tensor)

#     # 5. 处理检测结果并可视化
#     for result in results:
#         # 将结果绘制到原始图像上
#         annotated_img = result.plot()
        
#         # 转换回BGR格式用于OpenCV显示
#         annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
#         # 显示结果
#         cv2.imshow("YOLOv11 Detection", annotated_img)
#         cv2.waitKey(0)  # 等待按键
#         cv2.destroyAllWindows()
        
#         # 保存结果（可选）
#         output_path = "detection_result.jpg"
#         cv2.imwrite(output_path, annotated_img)
#         print(f"检测结果已保存至: {output_path}")

# if __name__ == "__main__":
#     # 替换为你的图像路径
#     image_path = "test_imgs\old.png"
#     model_path="models\yolo11n.pt"
#     main(image_path,model_path)
    


from ultralytics import YOLO
import cv2

# ===================== 加载官方 YOLOv3 模型 =====================
model = YOLO("yolov3.pt")  # 自动下载官方权重

# ===================== 推理（图片 / 视频 / 摄像头） =====================
# 图片推理
img_path = "test_imgs/car.jpg"
results = model.predict(
    source=img_path,      # 输入
    conf=0.25,           # 置信度
    iou=0.45,            # NMS IOU
    device="cuda",       # 使用 GPU
    save=True,           # 自动保存带框图片
    verbose=True         # 打印信息
)

# ===================== 手动获取结果（可选） =====================
for result in results:
    img = cv2.imread(img_path)
    
    # 框、类别、置信度
    boxes = result.boxes.xyxy.cpu().numpy()  # x1 y1 x2 y2
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()
    class_names = result.names

    # 画框
    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imwrite("yolov3_det.jpg", img)
print("推理完成，已保存 yolov3_det.jpg")