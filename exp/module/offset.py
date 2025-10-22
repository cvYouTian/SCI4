import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.font_manager as fm

# 查找系统中可用的中文字体
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in fonts if any(keyword in f for keyword in ['SimHei', 'Microsoft', 'Chinese', 'WenQuanYi'])]
# 设置字体
if chinese_fonts:
    plt.rcParams['font.family'] = [chinese_fonts[0]]

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class LocalSimGuidedSampler(nn.Module):

    def __init__(self, in_channels, scale=2, groups=4, kernel_size=1):
        super(LocalSimGuidedSampler, self).__init__()
        out_channels = 2 * groups * scale ** 2

        self.scale = scale
        self.groups = groups
        # 定义特征的方向
        self.offset = nn.Conv2d(3 ** 2 - 1, out_channels, kernel_size, padding=kernel_size // 2)
        normal_init(self.offset, std=0.001)

        # 定义特征的大小
        self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        constant_init(self.direct_scale, val=0.)

        out_channels = 2 * groups

        # 定义特征方向
        self.hr_offset = nn.Conv2d(3 ** 2 - 1, out_channels, kernel_size, padding=kernel_size // 2)

        normal_init(self.hr_offset, std=0.001)

        # 定义特征大小
        self.hr_direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        constant_init(self.hr_direct_scale, val=0.)

        # 是否作正则化
        self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    @staticmethod
    def direction_feat(input_tensor, k=3):
        # 计算像素中心和它的8邻域的余弦相似性
        B, C, H, W = input_tensor.shape
        unfold_tensor = F.unfold(input_tensor, k, padding=k // 2)
        unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)
        # 计算中心点和相邻点的余弦相似度
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
        # 去除中心的特征点
        similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)
        # 对结果进行reshape
        similarity = similarity.view(B, k * k - 1, H, W)

        return similarity

    def sample(self, x, offset, scale=None):
        if scale is None: scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = torch.stack(torch.meshgrid([coords_w, coords_h])). \
            transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)

    def get_offset(self, hf, lf, hf_sim, lf_sim):
        # 对应文章中的公式10，这里乘上一个动态
        offset = (self.offset(lf_sim) + F.pixel_unshuffle(self.hr_offset(hf_sim), self.scale)) * \
                 (self.direct_scale(lf) + F.pixel_unshuffle(self.hr_direct_scale(hf),
                                                            self.scale)).sigmoid() + self.init_pos

        return offset

    def forward(self, hf, lf, feat2sample):
        # 这里feat2sample可能是常规融合后的特征
        # 这里只是对输入进行归一化，没有其他的作用

        # f1 = self.norm_hr(f1)
        # f2 = self.norm_hr(f2)

        # 这里得到两个方向的特征矩阵【B， 8, H， W】
        hf_sim = self.direction_feat(hf)
        lf_sim = self.direction_feat(lf)
        offset = self.get_offset(hf, lf, hf_sim, lf_sim)

        return self.sample(feat2sample, offset)


# ============= 可视化函数 =============

def visualize_feature_maps(tensor, title="Feature Maps", max_channels=16):
    """
    可视化特征图
    Args:
        tensor: 输入张量 [B, C, H, W]
        title: 图标题
        max_channels: 最大显示通道数
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取第一个batch

    C, H, W = tensor.shape
    channels_to_show = min(C, max_channels)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(channels_to_show):
        feature_map = tensor[i].detach().cpu().numpy()
        im = axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # 隐藏多余的子图
    for i in range(channels_to_show, 16):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_similarity_maps(sim_tensor, title="Similarity Maps"):
    """
    可视化相似性图
    Args:
        sim_tensor: 相似性张量 [B, 8, H, W] (8个邻域方向)
        title: 图标题
    """
    if sim_tensor.dim() == 4:
        sim_tensor = sim_tensor[0]  # 取第一个batch

    directions = ['左上', '上', '右上', '左', '右', '左下', '下', '右下']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        similarity_map = sim_tensor[i].detach().cpu().numpy()
        im = axes[i].imshow(similarity_map, cmap='RdYlBu', vmin=-1, vmax=1)
        axes[i].set_title(f'{directions[i]}方向相似性')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_offset_field(offset, title="Offset Field"):
    """
    可视化偏移场
    Args:
        offset: 偏移张量 [B, 2*groups*scale^2, H, W]
        title: 图标题
    """
    if offset.dim() == 4:
        offset = offset[0]  # 取第一个batch

    # 重新组织偏移量维度
    C, H, W = offset.shape
    offset = offset.view(2, -1, H, W)  # [2, groups*scale^2, H, W]

    # 取第一组偏移量进行可视化
    dx = offset[0, 0].detach().cpu().numpy()  # x方向偏移
    dy = offset[1, 0].detach().cpu().numpy()  # y方向偏移

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # X方向偏移
    im1 = axes[0].imshow(dx, cmap='RdBu')
    axes[0].set_title('X方向偏移')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Y方向偏移
    im2 = axes[1].imshow(dy, cmap='RdBu')
    axes[1].set_title('Y方向偏移')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # 偏移矢量场
    y, x = np.mgrid[0:H:4, 0:W:4]  # 降采样显示矢量
    dx_sub = dx[::4, ::4]
    dy_sub = dy[::4, ::4]

    axes[2].quiver(x, y, dx_sub, dy_sub, angles='xy', scale_units='xy', scale=1)
    axes[2].set_title('偏移矢量场')
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def compare_input_output(input_lr, output_hr, title="Input vs Output Comparison"):
    """
    比较输入和输出特征
    Args:
        input_lr: 低分辨率输入 [B, C, H, W]
        output_hr: 高分辨率输出 [B, C, H*scale, W*scale]
        title: 图标题
    """
    if input_lr.dim() == 4:
        input_lr = input_lr[0, 0]  # 取第一个batch的第一个通道
    if output_hr.dim() == 4:
        output_hr = output_hr[0, 0]  # 取第一个batch的第一个通道

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 输入低分辨率
    im1 = axes[0].imshow(input_lr.detach().cpu().numpy(), cmap='viridis')
    axes[0].set_title('输入 (低分辨率)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # 输出高分辨率
    im2 = axes[1].imshow(output_hr.detach().cpu().numpy(), cmap='viridis')
    axes[1].set_title('输出 (高分辨率)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # 双线性上采样对比
    upsampled = F.interpolate(input_lr.unsqueeze(0).unsqueeze(0),
                              size=output_hr.shape, mode='bilinear', align_corners=False)
    im3 = axes[2].imshow(upsampled[0, 0].detach().cpu().numpy(), cmap='viridis')
    axes[2].set_title('双线性上采样对比')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_model_stats(model, hf, lf, feat2sample):
    """
    分析模型统计信息
    """
    model.eval()
    with torch.no_grad():
        # 前向传播
        output = model(hf, lf, feat2sample)

        # 计算统计信息
        hf_sim = model.direction_feat(hf)
        lf_sim = model.direction_feat(lf)
        offset = model.get_offset(hf, lf, hf_sim, lf_sim)

        print("=== 模型统计信息 ===")
        print(f"输入形状:")
        print(f"  高分辨率特征 (hf): {hf.shape}")
        print(f"  低分辨率特征 (lf): {lf.shape}")
        print(f"  待采样特征 (feat2sample): {feat2sample.shape}")

        print(f"\n中间结果形状:")
        print(f"  高分辨率相似性 (hf_sim): {hf_sim.shape}")
        print(f"  低分辨率相似性 (lf_sim): {lf_sim.shape}")
        print(f"  偏移场 (offset): {offset.shape}")

        print(f"\n输出形状:")
        print(f"  最终输出: {output.shape}")

        print(f"\n数值统计:")
        print(f"  hf_sim 范围: [{hf_sim.min():.4f}, {hf_sim.max():.4f}]")
        print(f"  lf_sim 范围: [{lf_sim.min():.4f}, {lf_sim.max():.4f}]")
        print(f"  offset 范围: [{offset.min():.4f}, {offset.max():.4f}]")
        print(f"  output 范围: [{output.min():.4f}, {output.max():.4f}]")

        return {
            'output': output,
            'hf_sim': hf_sim,
            'lf_sim': lf_sim,
            'offset': offset
        }


def comprehensive_test():
    """
    综合测试函数
    """
    print("=== LocalSimGuidedSampler 综合测试 ===")

    # 创建模型
    model = LocalSimGuidedSampler(64, scale=2, groups=4, kernel_size=1)

    # 创建测试数据
    batch_size = 2
    channels = 64
    lr_size = 8
    hr_size = 16

    lf = torch.randn(batch_size, channels, lr_size, lr_size)
    hf = torch.randn(batch_size, channels, hr_size, hr_size)
    feat2sample = torch.randn(batch_size, channels, lr_size, lr_size)

    # 分析模型
    results = analyze_model_stats(model, hf, lf, feat2sample)

    print("\n=== 开始可视化 ===")

    # 1. 可视化输入特征
    print("1. 可视化输入特征...")
    visualize_feature_maps(lf, "低分辨率输入特征 (LF)")
    visualize_feature_maps(hf, "高分辨率输入特征 (HF)")

    # 2. 可视化相似性图
    print("2. 可视化相似性图...")
    visualize_similarity_maps(results['hf_sim'], "高分辨率相似性图")
    visualize_similarity_maps(results['lf_sim'], "低分辨率相似性图")

    # 3. 可视化偏移场
    print("3. 可视化偏移场...")
    visualize_offset_field(results['offset'], "偏移场可视化")

    # 4. 比较输入输出
    print("4. 比较输入输出...")
    compare_input_output(feat2sample, results['output'], "输入输出对比")

    print("=== 测试完成 ===")


if __name__ == "__main__":
    # 运行综合测试
    comprehensive_test()