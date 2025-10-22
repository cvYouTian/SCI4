# 这是一个特征可视化工具
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from exp.module.Gaussian import LoGFilter
from exp.module.edge_extr import Get_gradient_nopadding, Get_gradientmask_nopadding
from exp.module.CDConv import CDC_conv, Conv


def show_feature(out, ch):
    out_cpu = out.cpu()
    feature_map = out_cpu.detach().numpy()
    # [N， C, H, W] -> [H, W， C]
    im = np.squeeze(feature_map)
    im = np.transpose(im, [1, 2, 0])
    for c in range(ch):
        ax = plt.subplot(3, 6, c + 1)
        plt.axis('off')  # 不显示坐标轴
        # [H, W, C]
        # plt.imshow(im[:, :, c], cmap=plt.get_cmap('Blues'))
        plt.imshow(im[:, :, c])
    plt.show()


if __name__ == '__main__':
    image_path = "/home/youtian/Documents/pro/pyCode/CFFNet-V2/data/dataset/IRSTD-1k/images/XDU16.png"  # 替换为你的图像路径
    img = Image.open(image_path).convert('RGB')  # 确保是RGB格式

    # 2. 转换为PyTorch张量
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

    # 3. 初始化LoG滤波器
    # log_filter = LoGFilter(
    #     in_c=3,
    #     out_c=3,  # 保持通道数不变
    #     kernel_size=9,
    #     sigma=1.0,
    #     norm_layer=dict(type='BN', requires_grad=True),  # 或你使用的其他norm
    #     act_layer=nn.ReLU  # 或你使用的其他激活
    # )

    # 原始的边缘特征提取网络
    grad = Get_gradient_nopadding()
    grad_mask = Get_gradientmask_nopadding()
    cdc_conv = CDC_conv(3, 16)
    conv = Conv(3, 16)

    # 4. 处理图像
    with torch.no_grad():
        # output = log_filter(img_tensor)
        # output = grad(img_tensor)
        output = cdc_conv(img_tensor)
        # output = conv(img_tensor)

    show_feature(output, 16)

    # 5. 可视化结果
    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.title("Original Image")
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # # 取第一个输出通道（或合并所有通道）
    # result = output.squeeze().permute(1, 2, 0).numpy()
    # # result = (result - result.min()) / (result.max() - result.min())  # 归一化到[0,1]
    # plt.imshow(result[:,:,1])  # 显示第一个通道或使用彩色
    # plt.title("LoG Filtered")
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()