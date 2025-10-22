import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Get_gradient_nopadding(nn.Module):
    """提取边缘特征"""

    def __init__(self, gpu=False):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

        if gpu:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        else:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cpu()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cpu()

    def forward(self, x):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)
        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        return torch.cat([x0, x1, x2], dim=1)


class Get_gradientmask_nopadding(nn.Module):
    """提取mask的边缘特征"""
    def __init__(self, gpu=False):
        super(Get_gradientmask_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

        if torch.cuda.is_available():
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        else:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cpu()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cpu()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        re = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        print(re.max())
        re = (re > 0.5).float()

        return re


def process_image(image_path):
    """处理图像并显示边缘检测结果"""

    # 1. 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. 转换为tensor
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    # 3. 移动到设备（如果有GPU的话）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)

    # 4. 边缘检测
    detector = Get_gradient_nopadding()
    with torch.no_grad():
        edge_result = detector(image_tensor[:, 0:1, :,:])
        print(edge_result.shape)
        edge_result = torch.sigmoid(edge_result)

    # 5. 转换回numpy
    edge_image = (edge_result.squeeze().cpu().numpy() > 0.5)
    edge_uint8 = (edge_image * 255).astype(np.uint8)  # 缩放到0-255并转为整数

    # 关键修改：将灰度数组复制到RGB三个通道
    edge_rgb = np.stack([edge_uint8] * 3, axis=-1)  # 形状从 (H,W) 变为 (H,W,3)
    edge_pil = Image.fromarray(edge_rgb, "RGB")  # 模式改为RGB
    # edge_pil.save("edge_rgb.png")

    # 6. 显示结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edge_image, cmap='gray')
    plt.title('边缘检测结果')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 修改这里的路径为你的图片路径
    image_path = "/home/youtian/Documents/pro/pyCode/CFFNet-V2/data/dataset/IRSTD-1k/images/XDU14.png"
    process_image(image_path)