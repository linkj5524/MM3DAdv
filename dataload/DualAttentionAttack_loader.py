

import os
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def euler_to_matrix(pitch, yaw, roll):
    """CARLA: yaw(z), pitch(y), roll(x)"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])

    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])

    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    return Rz @ Ry @ Rx

class MyDataset(Dataset):
    def __init__(self, data_dir, img_size, distence=None, mask_dir=''):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        
        # 距离过滤（保持不变）
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']
                cam_trans = data['cam_trans']

                cam_trans[0][0] += veh_trans[0][0]
                cam_trans[0][1] += veh_trans[0][1]
                cam_trans[0][2] += veh_trans[0][2]
                veh_trans[0][2] += 0.2

                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                if dis <= distence:
                    self.files.append(file)
                    
        print("有效样本数量:", len(self.files))
        self.img_size = img_size
        self.mask_dir = mask_dir
        self.scale = 0.41 # CARLA -> 渲染器 固定缩放，0.41

    # def __getitem__(self, index):
    #     file = os.path.join(self.data_dir, self.files[index])
    #     data = np.load(file)
        
    #     # ====================== 图像 + Mask 保持不变 ======================
    #     img = data['img']
    #     img = img[:, :, ::-1]
    #     img = cv2.resize(img, (self.img_size, self.img_size))
    #     img = img.astype(np.float32) / 255.0
    #     img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

    #     mask_file = os.path.join(self.mask_dir, self.files[index][:-4] + '.png')
    #     mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    #     mask = cv2.resize(mask, (self.img_size, self.img_size))
    #     mask = (mask > 127).astype(np.float32)
    #     mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    #     # ====================== 核心：相机相对车辆坐标系坐标 ======================
    #     cam_pose = data['cam_trans']       # [位置, 旋转]
    #     veh_pose = data['veh_trans']       # [位置, 旋转]

    #     cam_world_pos = cam_pose[0].copy()
    #     veh_world_pos = veh_pose[0].copy()

    #     # 1. 相机在世界坐标系下的位置
    #     cam_world_pos[0] += veh_world_pos[0]
    #     cam_world_pos[1] += veh_world_pos[1]
    #     cam_world_pos[2] += veh_world_pos[2]

    #     # 2. 车辆抬高 0.2m
    #     veh_world_pos[2] += 0.2

    #     # 3. 计算【相机相对于车辆】的世界偏移
    #     dx = cam_world_pos[0] - veh_world_pos[0]
    #     dy = cam_world_pos[1] - veh_world_pos[1]
    #     dz = cam_world_pos[2] - veh_world_pos[2]

    #     # 4. 车辆朝向（yaw），将世界坐标 旋转到 车辆车头坐标系
    #     veh_yaw = math.radians(veh_pose[1][1])
    #     cos_y = math.cos(veh_yaw)
    #     sin_y = math.sin(veh_yaw)

    #     # 坐标变换：世界 → 车辆局部坐标系（车头=X正）
    #     local_x = dx * cos_y + dy * sin_y
    #     local_y = -dx * sin_y + dy * cos_y
    #     local_z = dz

    #     # 5. 缩放（和FCA/DAS完全一致：0.41）
    #     rel_x = local_x * self.scale
    #     rel_y = local_y * self.scale
    #     rel_z = local_z * self.scale

    #     # 最终：相机在【车辆车头坐标系】下的相对坐标 (x,y,z)
    #     cam_relative_pos = torch.tensor([rel_x, rel_y, rel_z], dtype=torch.float32)

    #     # ======================================================================

    #     return index, img, mask, cam_relative_pos



    # def __getitem__(self, index):
    #     file = os.path.join(self.data_dir, self.files[index])
    #     data = np.load(file)

    #     # ====================== 图像 + Mask ======================
    #     img = data['img']
    #     img = img[:, :, ::-1]
    #     img = cv2.resize(img, (self.img_size, self.img_size))
    #     img = img.astype(np.float32) / 255.0
    #     img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

    #     mask_file = os.path.join(self.mask_dir, self.files[index][:-4] + '.png')
    #     mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    #     mask = cv2.resize(mask, (self.img_size, self.img_size))
    #     mask = (mask > 127).astype(np.float32)
    #     mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    #     # ====================== 读取位姿（强制 float64 高精度） ======================
    #     cam_pose = data['cam_trans']
    #     veh_pose = data['veh_trans']

    #     cam_pos = cam_pose[0].astype(np.float64)  # 相机相对偏移
    #     cam_rot = cam_pose[1].astype(np.float64)  # 相机旋转 [pitch, yaw, roll]

    #     veh_pos = veh_pose[0].astype(np.float64)  # 车辆世界位置
    #     veh_rot = veh_pose[1].astype(np.float64)  # 车辆旋转 [pitch, yaw, roll]

    #     # 相机世界位置 = 车辆世界位置 + 相机偏移
    #     cam_world_x = veh_pos[0] + cam_pos[0]
    #     cam_world_y = veh_pos[1] + cam_pos[1]
    #     cam_world_z = veh_pos[2] + cam_pos[2]

    #     # 车辆世界位置（抬高 0.2m）
    #     veh_world_x = veh_pos[0]
    #     veh_world_y = veh_pos[1]
    #     veh_world_z = veh_pos[2] + 0.2

    #     # 世界空间偏移
    #     dx = cam_world_x - veh_world_x
    #     dy = cam_world_y - veh_world_y
    #     dz = cam_world_z - veh_world_z

    #     # ====================== ✅ 核心修正：车辆朝向 + 相机姿态 双旋转 ======================
    #     # 车辆朝向 YAW（CARLA 世界 → 车辆局部）
    #     veh_yaw = math.radians(veh_rot[1])
    #     cy = math.cos(veh_yaw)
    #     sy = math.sin(veh_yaw)

    #     # 第一步：世界坐标 → 车辆局部坐标
    #     vx = dx * cy + dy * sy
    #     vy = -dx * sy + dy * cy
    #     vz = dz

    #     # 第二步：相机自身旋转（你给的相机是俯视 -90°  Pitch！必须加上！）
    #     cam_pitch = math.radians(cam_rot[0])
    #     cp = math.cos(cam_pitch)
    #     sp = math.sin(cam_pitch)

    #     # 相机俯视旋转：修正上下方向
    #     final_x = vx
    #     final_y = vy * cp - vz * sp
    #     final_z = vy * sp + vz * cp

    #     # 缩放
    #     rel_x = final_x * self.scale
    #     rel_y = final_y * self.scale
    #     rel_z = final_z * self.scale

    #     cam_relative_pos = torch.tensor([rel_x, rel_y, rel_z], dtype=torch.float32)

    #     return index, img, mask, cam_relative_pos




    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file)

        # ====================== 图像 ======================
        img = data['img']
        img = img[:, :, ::-1]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # ====================== Mask ======================
        mask_file = os.path.join(self.mask_dir, self.files[index][:-4] + '.png')
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        # ====================== 位姿 ======================
        cam_trans = data['cam_trans'].astype(np.float64)
        veh_trans = data['veh_trans'].astype(np.float64)


        rel_pos = cam_trans[0]
        rel_pos[2] += 0.2  # 车辆抬高 0.2m
        # ====================== 世界 → 车坐标 ======================

        veh_pitch   = math.radians(veh_trans[1][0])
        veh_yaw = math.radians(veh_trans[1][1])
        veh_roll  = math.radians(veh_trans[1][2])

        # 计算三角函数
        cy, sy = math.cos(veh_yaw), math.sin(veh_yaw)
        cp, sp = math.cos(veh_pitch), math.sin(veh_pitch)
        cr, sr = math.cos(veh_roll), math.sin(veh_roll)

        # Z轴旋转（yaw）
        Rz = np.array([
            [ cy, -sy, 0],
            [ sy,  cy, 0],
            [  0,   0, 1]
        ])

        # Y轴旋转（pitch）
        Ry = np.array([
            [ cp, 0, sp],
            [  0, 1,  0],
            [-sp, 0, cp]
        ])

        # X轴旋转（roll）
        Rx = np.array([
            [1,  0,   0],
            [0, cr, -sr],
            [0, sr,  cr]
        ])

        # 车辆 → 世界
        R = Rz @ Ry @ Rx

        # ====================== 世界 → 车（关键） ======================
        R_inv = R.T  # 正交矩阵，逆 = 转置

        # 坐标变换
        rel_pos_vehicle = R_inv @ rel_pos

        # # ====================== CARLA → PyTorch3D ======================
        x_final , y_final , z_final = rel_pos_vehicle * self.scale
        y_final=-y_final
        # ====================== 相机旋转 ======================
        cam_rot_world = cam_trans[1]  # pitch, yaw, roll（角度）

        pitch = math.radians(cam_rot_world[0])
        yaw   = math.radians(cam_rot_world[1])
        roll  = math.radians(cam_rot_world[2])

        # 先减去车辆 yaw（转到车坐标系）
        yaw_rel = yaw - veh_yaw
        pitch_rel = pitch
        roll_rel  = roll

        # 转回角度（保持接口一致）
        pitch_deg = math.degrees(pitch_rel)
        yaw_deg   = math.degrees(yaw_rel)
        roll_deg  = math.degrees(roll_rel)

        cam_pos_tensor = torch.tensor(
            [x_final, y_final, z_final],
            dtype=torch.float32
        )

        cam_rot_tensor = torch.tensor(
            [pitch_deg, yaw_deg, roll_deg],
            dtype=torch.float32
        )

        print(f"Camera position: {cam_trans}\n")
        print(f"Vehicle position: {veh_trans}\n")

        print(f"Camera rotation: {cam_pos_tensor}\n")
        print(f"Relative position (CARLA): {cam_rot_tensor}\n")
        # 转化为标准，x,y,z,右手系，车头朝x。
        # 实际车头朝x，Z是左右，y是上下


        # cam_pos_tensor = torch.stack([
        #     cam_pos_tensor[0],   # → X
        #     cam_pos_tensor[2],   # → Y
        #     -cam_pos_tensor[1],   # → Z
        # ])

        # # ================= 角度转换 =================
        # pitch, yaw, roll = cam_rot_tensor

        # cam_rot_tensor = torch.stack([
        #     roll,        # 新 pitch（绕X）
        #     pitch,       # 新 yaw（绕Y）
        #     -yaw         # 新 roll（绕Z，注意负号）
        # ])

        return index, img, mask, cam_pos_tensor, cam_rot_tensor







    def __len__(self):
        return len(self.files)