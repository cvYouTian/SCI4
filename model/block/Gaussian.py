import torch
import torch.nn as nn
# from mmcv.cnn import build_norm_layer
import math


class Conv_Extra(nn.Module):
    def __init__(self, channel, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   nn.BatchNorm2d(channel))

    def forward(self, x):
        out = self.block(x)
        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super(Gaussian, self).__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = norm_layer(dim)
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


# class LoGFilter(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
#         super(LoGFilter, self).__init__()
#         # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
#         self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
#         """创建高斯-拉普拉斯核"""
#         # 初始化二维坐标
#         ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
#         xx, yy = torch.meshgrid(ax, ax)
#         # 计算高斯-拉普拉斯核
#         kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (2 * math.pi * sigma ** 4) * torch.exp(
#             -(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
#
#         # 归一化
#         kernel = kernel - kernel.mean()
#         kernel = kernel / kernel.sum()
#         log_kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
#         self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
#                              groups=out_c, bias=False)
#         self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
#         self.act = act_layer()
#         self.norm1 = build_norm_layer(norm_layer, out_c)[1]
#         self.norm2 = build_norm_layer(norm_layer, out_c)[1]
#
#     def forward(self, x):
#         # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
#         x = self.conv_init(x)  # x = [B, C/4, H, W]
#         LoG = self.LoG(x)
#         LoG_edge = self.act(self.norm1(LoG))
#         x = self.norm2(x + LoG_edge)
#         return x
