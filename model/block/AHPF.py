import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    """应用动态滤波器"""
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    assert m_h == up * h
    assert m_w == up * w
    pad = kernel_size // 2

    # 对输入进行padding
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')

    # 展开邻域特征
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)

    if up > 1:
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')

    # 应用动态滤波器
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)

    return res


def hamming2D(M, N):
    """生成二维Hamming窗"""
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d


class AdaptiveHighPassFilter(nn.Module):
    """独立的自适应高通滤波器模块

    Args:
        in_channels: 输入通道数
        kernel_size: 高通滤波器的核大小，默认为3
        encoder_kernel: 生成器的卷积核大小，默认为3
        encoder_dilation: 生成器的膨胀率，默认为1
        groups: 滤波器组数，默认为1
        hamming_window: 是否使用Hamming窗进行正则化，默认为True
        residual: 是否使用残差连接，默认为True
    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 groups=1,
                 hamming_window=True,
                 residual=True):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.groups = groups
        self.residual = residual

        # 高通滤波器生成器
        self.filter_generator = nn.Conv2d(
            in_channels,
            kernel_size ** 2 * groups,
            encoder_kernel,
            padding=int((encoder_kernel - 1) * encoder_dilation / 2),
            dilation=encoder_dilation,
            groups=1
        )

        # Hamming窗用于正则化
        self.hamming_window = hamming_window
        if self.hamming_window:
            self.register_buffer('hamming',
                                 torch.FloatTensor(hamming2D(kernel_size, kernel_size))[None, None,])
        else:
            self.register_buffer('hamming', torch.FloatTensor([1.0]))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        xavier_init(self.filter_generator, distribution='uniform')
        normal_init(self.filter_generator, std=0.001)

    def kernel_normalizer(self, mask):
        """对生成的滤波器进行归一化

        将softmax应用于滤波器权重，确保权重和为1
        """
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(self.kernel_size ** 2))  # groups

        # 重塑并应用softmax
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)

        # 重塑回原始形状并应用Hamming窗
        mask = mask.view(n, mask_channel, self.kernel_size, self.kernel_size, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, self.kernel_size, self.kernel_size)
        mask = mask * self.hamming

        # 重新归一化确保权重和为1
        mask /= mask.sum(dim=(-1, -2), keepdims=True)

        # 转换回适合carafe的格式
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()

        return mask

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            enhanced_feat: 高频增强后的特征 [B, C, H, W]
        """
        # 生成高通滤波器参数
        filter_params = self.filter_generator(x)

        # 归一化得到低通滤波器
        lowpass_filter = self.kernel_normalizer(filter_params)

        # 应用低通滤波
        lowpass_feat = carafe(x, lowpass_filter, self.kernel_size, self.groups, 1)

        # 高通滤波 = 原始特征 - 低通滤波结果
        highpass_feat = x - lowpass_feat

        # 残差连接
        if self.residual:
            enhanced_feat = x + highpass_feat
        else:
            enhanced_feat = highpass_feat

        return enhanced_feat


# 测试代码
if __name__ == '__main__':
    # 创建AHPF模块
    ahpf = AdaptiveHighPassFilter(
        in_channels=128,
        kernel_size=3,
        encoder_kernel=3,
        residual=True
    )

    # 测试输入
    x = torch.randn(2, 128, 64, 64)

    # 前向传播
    enhanced_x = ahpf(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {enhanced_x.shape}")

    # 可视化高频增强效果
    with torch.no_grad():
        # 计算增强的高频分量
        high_freq = enhanced_x - x
        print(f"高频分量统计: mean={high_freq.mean():.4f}, std={high_freq.std():.4f}")

