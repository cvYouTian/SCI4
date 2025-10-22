
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F



# 替代简单的concat或add
class AdaptiveFusion(nn.Module):
    """
    自适应权重融合
    """
    def __init__(self, channels):
        super(AdaptiveFusion, self).__init__()
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_feat, edge_feat):
        combined = torch.cat([rgb_feat, edge_feat], dim=1)
        weights = self.weight_net(combined)
        return weights[:, 0:1] * rgb_feat + weights[:, 1:2] * edge_feat


class GatedFusion(nn.Module):
    """
    门控融合
    """
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, edge_feat):
        gate_weight = self.gate(torch.cat([rgb_feat, edge_feat], dim=1))
        return gate_weight * rgb_feat + edge_feat


class PyramidFusion(nn.Module):
    """
    金字塔融合
    """
    def __init__(self, channels):
        super().__init__()
        self.scales = [1, 2, 4]
        self.convs = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, 3, padding=1)
            for _ in self.scales
        ])

    def forward(self, rgb_feat, edge_feat):
        fused_feats = []
        for i, scale in enumerate(self.scales):
            if scale > 1:
                rgb_down = F.avg_pool2d(rgb_feat, scale)
                edge_down = F.avg_pool2d(edge_feat, scale)
                fused = self.convs[i](torch.cat([rgb_down, edge_down], dim=1))
                fused = F.interpolate(fused, size=rgb_feat.shape[2:])
            else:
                fused = self.convs[i](torch.cat([rgb_feat, edge_feat], dim=1))
            fused_feats.append(fused)
        return sum(fused_feats) / len(fused_feats)


# class SpatialAttentionFusion(nn.Module):
#     """空间注意力融合"""
#     def __init__(self, channels):
#         super().__init__()
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(2, 1, 7, padding=3),
#             nn.Sigmoid()
#         )
#
#     def forward(self, rgb_feat, edge_feat):
#         # 计算空间注意力
#         avg_pool = torch.mean(edge_feat, dim=1, keepdim=True)
#         max_pool = torch.max(edge_feat, dim=1, keepdim=True)[0]
#         spatial_att = self.spatial_att(torch.cat([avg_pool, max_pool], dim=1))
#
#         # 基于注意力调制边缘特征
#         modulated_edge = edge_feat * spatial_att
#         return rgb_feat + modulated_edge


class GaussianSmoothedFusion(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super(GaussianSmoothedFusion, self).__init__()
        self.gaussian_filter = self.get_gaussian_kernel(kernel_size, sigma)

    def get_gaussian_kernel(self, kernel_size, sigma):
        # 创建高斯卷积核
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[0, 0, i, j] = torch.exp(torch.tensor(
                    -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)
                ))
        return kernel / kernel.sum()

    def forward(self, rgb_feat, edge_feat):
        # 对边缘特征应用高斯平滑
        smoothed_edge = F.conv2d(edge_feat, self.gaussian_filter.to(edge_feat.device),
                                 padding=self.gaussian_filter.shape[-1] // 2, groups=edge_feat.shape[1])
        return rgb_feat + smoothed_edge

################################test module#######################################

def load_and_preprocess_image(image_path, size=(224, 224)):
    """加载并预处理图片"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加batch维度 [1, 3, H, W]


def visualize_results(rgb_img, edge_img, gate_weight, fused_output):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 转换为numpy格式用于显示
    def tensor_to_numpy(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).detach().numpy()

    # 第一行：输入图片和门控权重
    axes[0, 0].imshow(tensor_to_numpy(rgb_img))
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(tensor_to_numpy(edge_img))
    axes[0, 1].set_title('Edge Image')
    axes[0, 1].axis('off')

    # 门控权重可视化（取均值显示为灰度图）
    gate_avg = gate_weight.squeeze(0).mean(dim=0).detach().numpy()
    im = axes[0, 2].imshow(gate_avg, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('Gate Weight (avg)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # 第二行：各通道门控权重和融合结果
    gate_np = gate_weight.squeeze(0).detach().numpy()
    for i in range(3):
        axes[1, i].imshow(gate_np[i], cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Gate Channel {i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # 单独显示融合结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_numpy(fused_output))
    plt.title('Fused Output')
    plt.axis('off')

    # 显示融合结果与原图对比
    plt.subplot(1, 2, 2)
    # 简单的差异可视化
    diff = torch.abs(fused_output - rgb_img).squeeze(0).mean(dim=0).detach().numpy()
    plt.imshow(diff, cmap='hot')
    plt.title('Difference from RGB')
    plt.axis('off')
    plt.colorbar(fraction=0.046)

    plt.tight_layout()
    plt.show()


def test_with_images(rgb_path, edge_path):
    """使用真实图片测试GatedFusion"""
    print("加载图片...")

    # 如果没有指定图片路径，创建示例图片
    if rgb_path is None or edge_path is None:
        print("创建示例图片...")
        # 创建示例RGB图片 (渐变)
        rgb_img = torch.zeros(1, 3, 64, 64)
        for i in range(64):
            rgb_img[0, 0, i, :] = i / 64.0  # 红色渐变
            rgb_img[0, 1, :, i] = i / 64.0  # 绿色渐变
            rgb_img[0, 2, :, :] = 0.5  # 蓝色常数

        # 创建示例Edge图片 (棋盘格模式)
        edge_img = torch.zeros(1, 3, 64, 64)
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    edge_img[0, :, i:i + 8, j:j + 8] = 1.0
    else:
        # 加载真实图片
        rgb_img = load_and_preprocess_image(rgb_path)
        edge_img = load_and_preprocess_image(edge_path)

    print(f"图片形状: {rgb_img.shape}")

    # 创建融合模块
    fusion_module = GatedFusion(channels=3)
    fusion_module.eval()

    # 前向传播
    with torch.no_grad():
        fused_output = fusion_module(rgb_img, edge_img)
        gate_weight = fusion_module.gate(torch.cat([rgb_img, edge_img], dim=1))

    print(f"门控权重范围: [{gate_weight.min():.3f}, {gate_weight.max():.3f}]")
    print(f"融合输出范围: [{fused_output.min():.3f}, {fused_output.max():.3f}]")

    # 可视化结果
    visualize_results(rgb_img, edge_img, gate_weight, fused_output)

    return rgb_img, edge_img, gate_weight, fused_output


if __name__ == "__main__":
    # 使用示例图片测试（如果有真实图片，请修改路径）
    rgb_path = "/home/youtian/Documents/pro/pyCode/TGRS2025/exp/img/XDU146.png"  # 修改为你的RGB图片路径，如: "rgb_image.jpg"
    edge_path = "/home/youtian/Documents/pro/pyCode/TGRS2025/exp/img/XDU146_edge_mask.png"  # 修改为你的Edge图片路径，如: "edge_image.jpg"

    test_with_images(rgb_path, edge_path)