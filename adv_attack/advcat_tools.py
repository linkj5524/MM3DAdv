import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image, ImageDraw
from torch.nn.modules.activation import LeakyReLU, ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torch.nn.modules.linear import Linear
import torch.nn.init as init


import matplotlib.pyplot as plt


def hsv2rgb(hsv, saturate = None, color = None, bright = None):
    assert hsv.shape[1] == 3
    hsv_t = torch.zeros(hsv.shape).to(hsv.device)
    if saturate is not None:
        hsv_t[:,1] = torch.mul(hsv[:,1], saturate[1]) + torch.mul((1-hsv[:,1]),saturate[0])
    else:
        hsv_t[:,1] = hsv[:,1]
    if color is not None:
        hsv_t[:,0] = (hsv[:,0] * (color[1]-color[0]) + color[0]) % 360
    else:
        hsv_t[:,0] = hsv[:,0]*360
    if bright is not None:
        hsv_t[:,2] = hsv[:,2] * (bright[1]-bright[0]) + bright[0]
    else:
        hsv_t[:,2] = hsv[:,2]
    rgb = torch.zeros(hsv.shape).to(hsv.device)
    rgb_t = torch.zeros(hsv.shape).to(hsv.device)
    rgb[:,0] = hsv_t[:,2]
    rgb[:,1] = hsv_t[:,2] - hsv_t[:,1] * hsv_t[:,2] * torch.abs((hsv_t[:,0]/60)%2 -1)
    rgb[:,2] = hsv_t[:,2] - hsv_t[:,1]*hsv_t[:,2]
    # print(rgb)
    rgb_t[:,0] = torch.where(hsv_t[:,0] < 120, rgb[:,1], rgb[:,2])
    rgb_t[:,0] = torch.where(hsv_t[:,0]>= 240, rgb[:,1], rgb_t[:,0])
    rgb_t[:,0] = torch.where(hsv_t[:,0] < 60, rgb[:,0], rgb_t[:,0])
    rgb_t[:,0] = torch.where(hsv_t[:,0]>= 300, rgb[:,0], rgb_t[:,0])
    rgb_t[:,1] = torch.where(hsv_t[:,0] < 240, rgb[:,1], rgb[:,2])
    rgb_t[:,1] = torch.where(hsv_t[:,0] < 180, rgb[:,0], rgb_t[:,1])
    rgb_t[:,1] = torch.where(hsv_t[:,0] < 60, rgb[:,1], rgb_t[:,1])
    rgb_t[:,2] = torch.where(hsv_t[:,0]>= 120, rgb[:,1], rgb[:,2])
    rgb_t[:,2] = torch.where(hsv_t[:,0]>= 180, rgb[:,0], rgb_t[:,2])
    rgb_t[:,2] = torch.where(hsv_t[:,0]>= 300, rgb[:,1], rgb_t[:,2])
    return rgb_t


def blend(img1, img2):
    # Blend the second RGBA image to the first one. 
    # Note: This is not a symmetric function for the two images!
    # input images must be of the same size (N,4,H,W), RGBA.
    assert img1.shape[1] == 4 and img2.shape[1] == 4
    assert img1.shape[2] == img2.shape[2] and img1.shape[3] == img2.shape[3]
    img_blended_color = img1[:,:3,:,:]* img1[:,3,:,:](1-img2[:,3,:,:]) + img2[:,:3,:,:]*img2[:,3,:,:]
    img_blended_alpha = img1[:,3,:,:] + img2[:,3,:,:] - img1[:,3,:,:] * img2[:,3,:,:]
    img_blended = torch.cat([img_blended_color,img_blended_alpha],dim=1)
    return img_blended


def random_mask(figsize, num_geometry, prev_mask=None):
    # return a random 0/1 mask
    img_np = np.zeros([figsize,figsize,3])
    for i in range(num_geometry):
        img = Image.new("RGB", (figsize,figsize))
        draw = ImageDraw.Draw(img)
        x = np.random.randint(1,10)
        center = np.random.randint(0,figsize,[2])
        if x == 1:
            angle = np.random.randint(0,360,2)
            radius = np.random.randint(20,figsize/4)
            draw.chord([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius],angle[0], angle[1],fill=(1,1,1))
        if x == 2:
            radius = np.random.randint(20,figsize/4)
            draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius],fill=(1,1,1))
        if x == 3:
            offsets = np.random.randint(-100, 100, [3,2])
    #         print(center+offsets[0])
            draw.polygon([center[0]+offsets[0,0],center[1]+offsets[0,1], center[0]+offsets[1,0],center[1]+offsets[1,1]
                        ,center[0]+offsets[2,0],center[1]+offsets[2,1]],fill=(1,1,1))
        img_np += np.array(img)
    img_np = img_np % 2
    if prev_mask is None:
        prev_mask = np.ones(img_np.shape)
    img_np = torch.from_numpy(img_np * prev_mask)
    return img_np


def drawtriangles(origin_triangles,coordinates, fig_size):
    coordinates = coordinates.expand(origin_triangles.shape[0],-1,-1,-1).permute(1,2,0,3)
    triangles = origin_triangles*1.5*fig_size - 0.25*fig_size
    # print(triangles)
    s = (coordinates - triangles[:,0])*(triangles[:,1]-triangles[:,0])/torch.norm(triangles[:,1]-triangles[:,0])
    s = s.sum(dim = -1)
    l1 = s.le(0)*torch.norm(coordinates - triangles[:,0],dim=-1)
    l1 += s.ge(torch.norm(triangles[:,1]-triangles[:,0]))*torch.norm(coordinates - triangles[:,1],dim=-1)
    s = torch.norm(coordinates-triangles[:,0],dim=-1)**2-(s*s)
    s = s.ge(0)*(s)
    l1 += l1.le(1e-6)*torch.sqrt(s)

    s = (coordinates - triangles[:,1])*(triangles[:,2]-triangles[:,1])/torch.norm(triangles[:,2]-triangles[:,1])
    s = s.sum(dim = -1)
    l2 = s.le(0)*torch.norm(coordinates - triangles[:,1],dim=-1)
    l2 += s.ge(torch.norm(triangles[:,2]-triangles[:,1]))*torch.norm(coordinates - triangles[:,2],dim=-1)
    s = torch.norm(coordinates-triangles[:,1],dim=-1)**2-(s*s)
    s = s.ge(0)*(s)
    l2 += l2.le(1e-6)*torch.sqrt(s)


    s = (coordinates - triangles[:,2])*(triangles[:,0]-triangles[:,2])/torch.norm(triangles[:,0]-triangles[:,2])
    s = s.sum(dim = -1)
    l3 = s.le(0)*torch.norm(coordinates - triangles[:,2],dim=-1)
    l3 += s.ge(torch.norm(triangles[:,0]-triangles[:,2]))*torch.norm(coordinates - triangles[:,0],dim=-1)
    s = torch.norm(coordinates-triangles[:,2],dim=-1)**2-(s*s)
    s = s.ge(0)*(s)
    l3 += l3.le(1e-6)*torch.sqrt(s)

    distance = torch.min(l1,l2)
    distance = torch.min(distance,l3)

    t1 = (coordinates - triangles[:,0])
    t2 = (coordinates - triangles[:,1])
    t3 = (coordinates - triangles[:,2])
    q1 = (t1[...,0]*t2[...,1] - t1[...,1]*t2[...,0])
    q2 = (t2[...,0]*t3[...,1] - t2[...,1]*t3[...,0])
    q3 = (t3[...,0]*t1[...,1] - t3[...,1]*t1[...,0])
    q = (q1*q2).ge(0)*(q1*q3).ge(0) *(q2*q3).ge(0) * 2 -1
    prob = torch.sigmoid(distance**2 * q / 3)
    # print(prob.sum())
    return prob

def xor_mask(prob_map):
    prob_map = prob_map.sum(dim=-1)%2
    prob_xor = prob_map.ge(1+1e-10) * (2 - prob_map) + prob_map * prob_map.le(1)
    # print(prob_xor.max())
    return prob_xor

def xor_mask_color(prob_map,color):
    # print(prob_map)
    prob_map = prob_map.expand(3,-1,-1,-1).permute(1,2,3,0)
    color_map = (prob_map * color).sum(dim=-2) % 2
    color_xor = color_map.ge(1+1e-10) * (2 - color_map) + color_map * color_map.le(1)
    return color_xor

def drawcircles(original_circles, coordinates, fig_size, alpha=None):
    coordinates = coordinates.expand(original_circles.shape[0],-1,-1,-1).permute(1,2,0,3)
    circles = original_circles * fig_size
    dist = torch.norm(coordinates - circles[:,:2],dim=-1)
    dist = dist - circles[:,2]
    if alpha is not None:
        prob = torch.sigmoid(dist/alpha)
    else:
        prob = torch.sigmoid(dist)
    return prob

def drawcircles_with_blur(original_circles, coordinates, fig_size):
    # origin_circles: (num_circles x 4). (cx, cy, radius, blur)
    coordinates = coordinates.expand(original_circles.shape[0],-1,-1,-1).permute(1,2,0,3)
    circles = original_circles[:,:3] * fig_size
    dist = torch.norm(coordinates - circles[:,:2],dim=-1)
    dist = dist - circles[:,2]
    # circles[:,3] \in [0,1], clip them to [0.9,1].
    dist = dist*(original_circles[:,3]+1)/2
    prob = torch.sigmoid(dist)
    return prob


def drawcircles_fix_color(original_circles, coordinates, colors, fig_size_h, fig_size_w,blur=1):
    assert original_circles.shape[0] == colors.shape[0]
    coordinates = coordinates.expand(original_circles.shape[1],-1,-1,-1).permute(1,2,0,3)
    circle0 = original_circles[...,0]*fig_size_h
    circle1 = original_circles[...,1]*fig_size_w
    circles = torch.stack([circle0,circle1],dim=-1)
    dist_sum = torch.zeros([colors.shape[0],fig_size_h,fig_size_w]).to(coordinates.device)
    for color_idx in range(colors.shape[0]):
        dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
        # dist = dist / (circles[color_idx,:,2]+1)
        dist_sum[color_idx] = torch.exp(-dist/blur).sum(dim=-1)
        # print(dist_sum[color_idx])
    # print(dist_sum[0])
    dist_sum = dist_sum ** 2
    dist_sum = dist_sum/dist_sum.sum(dim=0)
    # print(dist_sum[0])
    color_map = torch.matmul(dist_sum.permute(1,2,0), colors).permute(2,0,1)
    # print(color_map)
    return color_map


# def prob_fix_color(original_circles, coordinates, colors, fig_size_h, fig_size_w,blur=1):
#     assert original_circles.shape[0] == colors.shape[0]
#     coordinates = coordinates.expand(original_circles.shape[1],-1,-1,-1).permute(1,2,0,3)
#     # circles = original_circles * fig_size_h
#     circle0 = original_circles[...,0]*fig_size_h
#     circle1 = original_circles[...,1]*fig_size_w
#     circles = torch.stack([circle0,circle1],dim=-1)
#     dist_sum = torch.zeros([colors.shape[0],fig_size_h,fig_size_w]).to(coordinates.device)
#     for color_idx in range(colors.shape[0]):
#         dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
#         # dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
#         # dist = dist / (circles[color_idx,:,2]+1)
#         dist_sum[color_idx] = torch.exp(-dist/blur).sum(dim=-1)
#         # print(dist_sum[color_idx])
#     # print(dist_sum[0])
#     dist_sum = dist_sum/dist_sum.sum(dim=0)
#     return dist_sum

def prob_fix_color(original_circles, coordinates, colors, fig_size_h, fig_size_w, blur=1):
    assert original_circles.shape[0] == colors.shape[0]
    coordinates = coordinates.expand(original_circles.shape[1], -1, -1, -1).permute(1, 2, 0, 3)
    
    # 安全处理尺寸缩放
    circle0 = original_circles[..., 0] * fig_size_h
    circle1 = original_circles[..., 1] * fig_size_w
    circles = torch.stack([circle0, circle1], dim=-1)
    
    dist_sum = torch.zeros([colors.shape[0], fig_size_h, fig_size_w], device=coordinates.device)
    
    for color_idx in range(colors.shape[0]):
        dist = torch.norm(coordinates - circles[color_idx, :, :2], dim=-1)
        dist_sum[color_idx] = torch.exp(-dist / blur).sum(dim=-1)
    
    #  核心修复：防止分母为 0，加极小值 eps
    sum_dist = dist_sum.sum(dim=0)
    eps = 1e-6
    sum_dist = torch.clamp(sum_dist, min=eps)  # 保证分母永远 >= eps
    
    # 归一化，永远不会出现 NaN
    dist_sum = dist_sum / sum_dist
    
    return dist_sum

# def gumbel_color_fix_seed(prob_map, seed, color, tau=0.3, type='gumbel'):
#     # print(prob_map.shape, seed.shape, color.shape)
#     if type == 'gumbel':
#         color_map = F.softmax((torch.log(prob_map) + seed)/tau, dim=-1)
#     elif type == 'determinate':
#         color_ind = (torch.log(prob_map) + seed).max(-1)[1]
#         color_map = F.one_hot(color_ind, prob_map.shape[-1]).to(prob_map)
#     else:
#         raise ValueError
#     tex = torch.matmul(color_map, color).unsqueeze(0)
#     return tex
def gumbel_color_fix_seed(prob_map, seed, color, tau=0.3, type='gumbel'):
    #  核心修复：把 prob_map 钳到极小值，防止 log(0) = -inf
    prob_map = torch.clamp(prob_map, min=1e-6, max=1.0)
    
    logits = torch.log(prob_map) + seed

    if type == 'gumbel':
        color_map = F.softmax(logits / tau, dim=-1)
    elif type == 'determinate':
        color_ind = logits.max(-1)[1]
        color_map = F.one_hot(color_ind, prob_map.shape[-1]).to(prob_map)
    else:
        raise ValueError(f"Unknown type: {type}")
    
    tex = torch.matmul(color_map, color).unsqueeze(0)
    return tex


def ctrl_loss(circles, fig_h, fig_w, sigma=40):
    circles = circles.repeat(circles.shape[1],1,1,1).permute(1,0,2,3)
    diff = circles - circles.permute(0,2,1,3)
    diff_ell2 = (diff[...,0] * diff[...,0]*fig_h*fig_h + diff[...,1] * diff[..., 1]*fig_w*fig_w)
    loss_c = torch.exp(-diff_ell2/(sigma**2)).mean() - 1/circles.shape[1]
    return loss_c
        

# colortransform
# class TotalVariation(nn.Module):
#     """计算 Total Variation Loss，和你代码里的 TV 完全对应"""
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         """
#         x: (1, H, W, 3)  你的纹理格式
#         """
#         h_diff = x[:, 1:, :, :] - x[:, :-1, :, :]
#         w_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
#         diff_sum = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
#         return diff_sum

class TotalVariation(nn.Module):
    """计算 Total Variation Loss，和你代码里的 TV 完全对应，修正求和逻辑"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: (1, H, W, 3)  你的纹理格式
        """
        h_diff = x[:, 1:, :, :] - x[:, :-1, :, :]  # shape: (1, H-1, W, 3)
        w_diff = x[:, :, 1:, :] - x[:, :, :-1, :]  # shape: (1, H, W-1, 3)
        
        # 修正1：逐通道计算L1范数之和，再对通道求和（遵循TV损失逻辑）
        h_loss = torch.sum(torch.abs(h_diff), dim=[1, 2])  # 对H-1、W维度求和，保留通道维度 (1, 3)
        w_loss = torch.sum(torch.abs(w_diff), dim=[1, 2])  # 对H、W-1维度求和，保留通道维度 (1, 3)
        channel_sum = torch.sum(h_loss + w_loss, dim=1)  # 对3个通道求和，得到每个batch的损失 (1,)
        
        # 修正2：归一化（除以差异总数，避免尺寸影响，约束力度稳定）
        # 差异总数 = 垂直差异数（(H-1)*W*3） + 水平差异数（H*(W-1)*3）
        batch_size, H, W, C = x.shape
        total_diff = (H - 1) * W * C + H * (W - 1) * C
        tv_loss = channel_sum / total_diff/batch_size  # 归一化后，损失值稳定在合理范围
        
        return tv_loss
    
class ColorTransform(nn.Module):
    def __init__(self, para_path):
        super(ColorTransform, self).__init__()
        file = np.load(para_path, allow_pickle=True)
        self.degree = file['d']
        weight = torch.from_numpy(file['weight'])
        bias = torch.from_numpy(file['bias'])
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def poly_feature(self, x, degree=None):
        if degree is None:
            degree = self.degree
        n = x.shape[1]
        feature = [x.clone()]
        index = list(range(n))
        for d in range(1, degree):
            new = []
            k = 0
            for i in range(n):
                new.append(x[:, i:i + 1] * feature[-1][:, index[i]:])
                index[i] = k
                k = k + new[-1].shape[1]
            new = torch.cat(new, 1)
            feature.append(new)
        feature = torch.cat(feature, 1)
        return feature

    def forward(self, x):
        f = self.poly_feature(x)
        f = f.transpose(1, -1)
        #     pred = (f.unsqueeze(1) * weight.unsqueeze(0)).sum(2) + bias
        pred = torch.matmul(f, self.weight) + self.bias
        pred = pred.transpose(1, -1)
        return pred
    
def reg_dist(x, dist='uniform', mode='cf', sample_num=200):
    i = 1
    t = x.new(size=[sample_num]).normal_()


    if dist == 'uniform':
        t_abs = t.abs()
        f_real = torch.sin(i * t_abs) / (t_abs + 1e-10)
        f_img = 2 * torch.sin(i * t_abs / 2).square() / (t_abs + 1e-10) * t.sign()
    else:
        raise NotImplementedError

    #     estimate f
    f_e_real = 1 / x.shape[-1] * torch.cos(t.unsqueeze(-1)*x.unsqueeze(-2)).sum(-1)
    f_e_img = 1 / x.shape[-1] * torch.sin(t.unsqueeze(-1)*x.unsqueeze(-2)).sum(-1)

    diff = (f_real - f_e_real)*(f_real - f_e_real) + (f_img - f_e_img)*(f_img - f_e_img)
    diff = diff  / (t * t + 1e-10) / (-t * t / 2).exp()
    diff = diff.sum(-1) / sample_num
    return diff

# def compute_regularization_losses(tex_dict, seeds_train_dict=None):
#     """
#     不依赖 args，内部写死所有超参数
#     """

#     # ===== 固定超参数 =====
#     TV_WEIGHT = 0.0
#     CTRL_WEIGHT = 1.0
#     SEED_WEIGHT = 0.0
#     RD_NUM = 200

#     device = next(iter(tex_dict.values()))['tex'].device

#     loss_tv = torch.zeros([], device=device)
#     loss_ctrl = torch.zeros([], device=device)
#     loss_seed = torch.zeros([], device=device)

#     tv_fn = TotalVariation()

#     num_mat = len(tex_dict)

#     for mat in tex_dict:
#         tex = tex_dict[mat]['tex']              # (1,H,W,3)
#         pointd = tex_dict[mat]['pointd']        # (C,N,3)
#         coords = tex_dict[mat]['coordinates']   # (H,W,2)

#         H, W = coords.shape[:2]

#         # ===== TV loss =====
#         if TV_WEIGHT > 0:
#             loss_tv += tv_fn(tex)

#         # ===== ctrl loss =====
#         if CTRL_WEIGHT > 0:
#             loss_ctrl += ctrl_loss(pointd, H, W)

#         # ===== seed loss =====
#         if SEED_WEIGHT > 0 and seeds_train_dict is not None:
#             seeds = seeds_train_dict[mat]
#             loss_seed += reg_dist(
#                 seeds.flatten(),
#                 sample_num=RD_NUM
#             )

#     # ===== 可选：防止材质数量影响loss尺度 =====
#     loss_tv /= num_mat
#     loss_ctrl /= num_mat
#     loss_seed /= num_mat

#     # ===== 加权 =====
#     loss_tv *= TV_WEIGHT
#     loss_ctrl *= CTRL_WEIGHT
#     loss_seed *= SEED_WEIGHT

#     loss_total = loss_tv + loss_ctrl + loss_seed

#     return loss_total, {
#         "tv_loss": loss_tv.detach(),
#         "ctrl_loss": loss_ctrl.detach(),
#         "seed_loss": loss_seed.detach(),
#         "reg_total": loss_total.detach()
#     }


def compute_regularization_losses(tex_dict, seeds_train_dict=None):
    """
    不依赖 args，内部写死所有超参数
    修复：全程保留计算图，仅日志信息 detach，不影响梯度回传
    """

    # ===== 固定超参数 =====
    TV_WEIGHT = 0.1
    CTRL_WEIGHT = 1.0
    SEED_WEIGHT = 0.0
    RD_NUM = 200

    device = next(iter(tex_dict.values()))['tex'].device

    # 初始化带有梯度追踪属性的标量 0
    loss_tv = torch.tensor(0.0, device=device, requires_grad=True)
    loss_ctrl = torch.tensor(0.0, device=device, requires_grad=True)
    loss_seed = torch.tensor(0.0, device=device, requires_grad=True)

    tv_fn = TotalVariation()

    num_mat = len(tex_dict)

    for mat in tex_dict:
        tex = tex_dict[mat]['tex']              # (1,H,W,3)
        pointd = tex_dict[mat]['pointd']        # (C,N,3)
        coords = tex_dict[mat]['coordinates']   # (H,W,2)

        H, W = coords.shape[:2]

        # ===== TV loss =====
        if TV_WEIGHT > 0:
            loss_tv = loss_tv + tv_fn(tex)  # 不用 +=，更稳定

        # ===== ctrl loss =====
        if CTRL_WEIGHT > 0:
            loss_ctrl = loss_ctrl + ctrl_loss(pointd, H, W)

        # ===== seed loss =====
        if SEED_WEIGHT > 0 and seeds_train_dict is not None:
            seeds = seeds_train_dict[mat]
            loss_seed = loss_seed + reg_dist(
                seeds.flatten(),
                sample_num=RD_NUM
            )

    # ===== 平均（保持计算图）=====
    loss_tv = loss_tv / num_mat
    loss_ctrl = loss_ctrl / num_mat
    loss_seed = loss_seed / num_mat

    # ===== 加权（保持计算图）=====
    loss_tv = loss_tv * TV_WEIGHT
    loss_ctrl = loss_ctrl * CTRL_WEIGHT
    loss_seed = loss_seed * SEED_WEIGHT

    # 总损失（可直接用于反向传播）
    loss_total = loss_tv + loss_ctrl + loss_seed

    # ========== 关键修复：只在返回日志时 detach ==========
    return loss_total, {
        "tv_loss": loss_tv,        # 仅日
        "ctrl_loss": loss_ctrl,    # 仅日志
        "seed_loss": loss_seed,    # 仅日志
        "reg_total": loss_total    # 仅日志
    }

class advcat_attack(object):
    def __init__(self, device, target_material_list, image_size, num_points,
                 lr=1e-2, lr_seed=1e-2, clamp_shift=0.1):
        self.clamp_shift = clamp_shift
        # ===== 基础 =====
        self.device = device
        self.tex_dict = {}   

        color_transform = ColorTransform('/root/autodl-tmp/adv_method/MM3DAdv/needed_data/color_transform_dim6.npz')
        self.color_transform = color_transform.to(device)

        num_colors = 4
        h, w = image_size

        # ===== 材质参数 =====
        optim_params = []
        
        for key_material in target_material_list:
            self.tex_dict[key_material] = {}

            # 坐标
            self.tex_dict[key_material]['coordinates'] = torch.stack(
                torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing='ij'   # 
                ), -1
            )

            # 可学习点
            pointd = torch.rand(
                [num_colors, num_points, 3],
                device=device,
                requires_grad=True
            )
            self.tex_dict[key_material]['pointd'] = pointd

            optim_params.append(pointd)  # ✅ 收集到优化器

        self.colors = torch.tensor([
                [0.12, 0.25, 0.10],  # 深墨绿
                [0.22, 0.38, 0.18],  # 草绿色
                [0.35, 0.42, 0.25],  # 橄榄黄
                [0.08, 0.15, 0.08]   # 暗墨绿色
            ], dtype=torch.float32, device=self.device)

        # ===== 优化器（点）=====
        self.optimizer = torch.optim.Adam(optim_params, lr=lr)

        # ===== seeds（多材质版本）=====
        self.seeds_train = {}
        self.seeds_fixed = {}

        seed_params = []

        for key_material in target_material_list:
            seeds_train = torch.zeros(
                size=[h, w, num_colors],
                device=device
            ).uniform_(clamp_shift, 1 - clamp_shift).requires_grad_()

            seeds_fixed = torch.zeros(
                size=[h, w, num_colors],
                device=device
            ).uniform_()

            self.seeds_train[key_material] = seeds_train
            self.seeds_fixed[key_material] = seeds_fixed

            seed_params.append(seeds_train)

        # ===== 优化器（seed）=====
        self.optimizer_seed = torch.optim.Adam(seed_params, lr=lr_seed)

        # ===== 平滑卷积核 =====
        k = 3
        k2 = k * k
        self.camouflage_kernel = nn.Conv2d(
            num_colors, num_colors, k, 1, k // 2
        ).to(device)

        self.camouflage_kernel.weight.data.zero_()
        self.camouflage_kernel.bias.data.zero_()

        for i in range(num_colors):
            self.camouflage_kernel.weight[i, i, :, :].data.fill_(1.0 / k2)


    def update_mesh(self, tau=0.3, type='gumbel', blur=1):
        """
        多材质版本 update_mesh
        输出写入 self.tex_dict[material]['tex']
        """

        for key_material in self.tex_dict:

            # ===== 取参数 =====
            pointd = self.tex_dict[key_material]['pointd']
            coords = self.tex_dict[key_material]['coordinates']
            seeds  = self.seeds_train[key_material]

            h, w = coords.shape[:2]

            # ===== prob map =====
            prob_map = prob_fix_color(
                pointd,
                coords,
                self.colors,
                h,
                w,
                blur=blur
            ).unsqueeze(0)   # (1, C, H, W)

            prob_map = self.camouflage_kernel(prob_map)

            prob_map = prob_map.squeeze(0).permute(1, 2, 0)  # (H, W, C)
            prob_map = torch.clamp(prob_map, min=1e-6, max=1.0)
            # ===== Gumbel noise =====
            gb = -(-(seeds + 1e-20).log() + 1e-20).log()

            # ===== 采样颜色 =====
            tex = gumbel_color_fix_seed(
                prob_map,
                gb,
                self.colors,
                tau=tau,
                type=type
            )  # (1, H, W, 3)

            # ===== 颜色空间变换 =====
            tex = self.color_transform(
                tex.permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)

            # ===== 写回 =====
            self.tex_dict[key_material]['tex'] = tex

        return self.tex_dict



    def step(self, det_loss, tau=0.3, type='gumbel', blur=1):
        """
        det_loss: 外部传入的检测对抗损失（模型前向得到的攻击损失）
        return: 总损失、各分项损失字典
        """
        # 1. 清空所有优化器梯度
        self.optimizer.zero_grad()
        self.optimizer_seed.zero_grad()

        # # 2. 前向：更新生成最新材质纹理
        # self.update_mesh(tau=tau, type=type, blur=blur)

        # 3. 计算所有正则化损失
        loss_reg, loss_info = compute_regularization_losses(
            tex_dict=self.tex_dict,
            seeds_train_dict=self.seeds_train
        )
        loss_tv = loss_info['tv_loss']
        loss_ctrl = loss_info['ctrl_loss']
        loss_seed = loss_info['seed_loss']

        # 4. 计算加权总损失 = 对抗损失 + 加权正则损失
        total_loss = self.weight_det * det_loss +loss_reg 

        # 5. 反向传播 计算所有变量梯度
        total_loss.backward()

        # 6. 第一步：优化 可学习点参数 pointd
        self.optimizer.step()

        # 7. 第二步：优化 可学习种子 seeds_train（完全对齐原版代码逻辑）
        # seed梯度缩放
        for mat in self.seeds_train:
            self.seeds_train[mat].grad /= self.seed_temp
        self.optimizer_seed.step()

        # 8. 参数值域裁剪（严格原版规则）
        # 1）材质控制点 pointd 限制 [0, 1]
        for mat in self.tex_dict:
            self.tex_dict[mat]['pointd'].data.clamp_(0.0, 1.0)
        # 2）颜色参数裁剪
        self.colors.data.clamp_(0.0, 1.0)
        # 3）训练种子限制边界
        for mat in self.seeds_train:
            self.seeds_train[mat].data.clamp_(self.clamp_shift, 1.0 - self.clamp_shift)

        # 返回所有损失值，方便打印、日志记录
        loss_all = {
            "det_loss": det_loss.detach(),
            "tv_loss": loss_tv,
            "ctrl_loss": loss_ctrl,
            "seed_loss": loss_seed,
            "reg_total": loss_reg.detach(),
            "total_loss": total_loss.detach()
        }
        return total_loss.item(), loss_all





