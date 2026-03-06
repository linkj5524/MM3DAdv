import cv2
import numpy as np
import torch
from PIL import Image

def image_canny_to_tensor(img_path, 
                          canny_low=5, 
                          canny_high=10,
                          blur_kernel=(3,3)):
    """
    输入图片路径，直接计算Canny边缘，返回0/1的二值Tensor
    
    参数:
        img_path (str): 图片文件路径
        canny_low (int): Canny边缘检测低阈值（默认5）
        canny_high (int): Canny边缘检测高阈值（默认10）
        blur_kernel (tuple): 高斯模糊核大小（默认(3,3)）
    
    返回:
        torch.Tensor: 形状为[H, W]的二值Tensor，值为0或1（1表示边缘）
    """
    # -------------------------- 1. 读取并预处理图片 --------------------------
    # 读取图片（BGR格式）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径是否正确：{img_path}")
    
    # 转换为灰度图（Canny检测基于灰度图）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪（减少边缘检测的噪声，避免伪边缘）
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # -------------------------- 2. 直接执行Canny边缘检测 --------------------------
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # -------------------------- 3. 转换为0/1 Tensor --------------------------
    # 将边缘图（255表示边缘，0表示非边缘）转为0/1的numpy数组
    edges_binary = (edges / 255).astype(np.float32)
    
    # 转换为PyTorch Tensor（形状[H, W]，值为0或1）
    edge_tensor = torch.from_numpy(edges_binary)
    
    return edge_tensor

# -------------------------- 测试函数 --------------------------
if __name__ == "__main__":
    # 测试示例（替换为你的图片路径）
    test_img_path = "./data/ocean.jpg"
    try:
        # 调用函数
        edge_tensor = image_canny_to_tensor(test_img_path,
                                            canny_low=50,
                                            canny_high=200,
                                            blur_kernel=(3,3))

        
        # 打印结果信息
        print(f"Tensor形状: {edge_tensor.shape}")
        print(f"Tensor值范围: [{edge_tensor.min().item()}, {edge_tensor.max().item()}]")
        print(f"Tensor数据类型: {edge_tensor.dtype}")
        print(f"边缘像素数量: {torch.sum(edge_tensor).item()}")
        
        # 可选：可视化结果
        import matplotlib.pyplot as plt
        plt.imshow(edge_tensor.numpy(), cmap='gray')
        plt.title("Canny Edge Result (0/1 Tensor)")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"执行出错: {e}")