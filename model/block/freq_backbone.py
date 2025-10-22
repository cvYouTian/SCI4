import torch
import torch.nn as nn
import torch.nn.functional as F
from model.block.faardown import DWTForward


class FrequencyFeatureExtractor(nn.Module):
    """
    频域特征提取主干网络，专门用于提取红外图像的高频特征（边缘特征）

    设计思路：
    1. 使用多尺度小波变换分解图像为不同频域子带
    2. 专门设计高频子带的特征提取网络
    3. 使用注意力机制增强边缘特征
    4. 多层次特征融合保持小目标信息
    """

    def __init__(self, in_channels=1, base_channels=64, num_levels=3, wavelet='db4'):
        super(FrequencyFeatureExtractor, self).__init__()

        self.num_levels = num_levels
        self.base_channels = base_channels

        # 初始特征提取
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 多级小波变换
        self.dwt_transforms = nn.ModuleList()
        for i in range(num_levels):
            self.dwt_transforms.append(DWTForward(J=1, wave=wavelet, mode='symmetric'))

        # 高频子带特征提取器
        self.high_freq_extractors = nn.ModuleList()
        for i in range(num_levels):
            channels = base_channels * (2 ** i)
            # 每个尺度的高频特征提取器
            self.high_freq_extractors.append(HighFreqSubbandExtractor(channels, channels*2))

        # 边缘增强注意力模块
        self.edge_attention = EdgeAttentionModule(base_channels)

        # 多尺度特征融合
        self.multiscale_fusion = MultiscaleFreqFusion(base_channels, num_levels)

    def forward(self, x):
        # 初始特征提取
        feat = self.input_conv(x)

        # 存储各尺度的高频特征
        high_freq_features = []
        low_freq_features = []

        current_feat = feat

        for i in range(self.num_levels):
            # 小波变换分解
            low, high_list = self.dwt_transforms[i](current_feat)

            # 高频子带特征提取 (HL, LH, HH)
            high_freq_feat = self.high_freq_extractors[i](high_list[0])
            high_freq_features.append(high_freq_feat)
            low_freq_features.append(low)

            # 下一层输入
            current_feat = low

        # 边缘注意力增强
        enhanced_feat = self.edge_attention(feat, high_freq_features[0])

        # 多尺度频域特征融合
        fused_feat = self.multiscale_fusion(high_freq_features, low_freq_features, enhanced_feat)

        return {
            'high_freq_features': high_freq_features,
            'low_freq_features': low_freq_features,
            'enhanced_feat': enhanced_feat,
            'fused_feat': fused_feat
        }


class FrequencyEdgeEnhancement(nn.Module):
    """频域边缘增强模块
    
    原理：
    - 使用DCT/DWT提取高频分量
    - 高频分量包含边缘信息
    - 增强高频，抑制低频背景
    """
    
    def __init__(self, channels, freq_type='dct'):
        super(FrequencyEdgeEnhancement, self).__init__()
        self.freq_type = freq_type
        self.channels = channels
        
        # 高频增强权重
        self.high_freq_weight = nn.Parameter(torch.ones(1))
        self.low_freq_weight = nn.Parameter(torch.ones(1) * 0.1)
        
        # 频域特征处理
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def dct_2d(self, x):
        """2D离散余弦变换"""
        # 简化版DCT实现
        N, C, H, W = x.shape
        
        # 使用FFT近似DCT
        x_freq = torch.fft.rfft2(x, norm='ortho')
        
        return x_freq
    
    def idct_2d(self, x_freq, size):
        """2D逆离散余弦变换"""
        x = torch.fft.irfft2(x_freq, s=size, norm='ortho')
        return x
    
    def forward(self, x):
        N, C, H, W = x.shape
        
        # 转到频域
        x_freq = self.dct_2d(x)
        
        # 分离高频和低频
        # 高频：边缘、细节（频谱外围）
        # 低频：背景、平滑区域（频谱中心）
        freq_h, freq_w = x_freq.shape[-2:]
        
        # 创建高频掩码（外围）
        mask_h = torch.ones_like(x_freq.real)
        mask_l = torch.zeros_like(x_freq.real)
        
        # 中心区域为低频
        center_h, center_w = freq_h // 4, freq_w // 4
        mask_l[:, :, :center_h, :center_w] = 1
        mask_h = mask_h - mask_l
        
        # 分离高低频
        high_freq = x_freq * mask_h
        low_freq = x_freq * mask_l
        
        # 增强高频，抑制低频
        enhanced_freq = high_freq * self.high_freq_weight + low_freq * self.low_freq_weight
        
        # 转回空域
        x_enhanced = self.idct_2d(enhanced_freq, (H, W))
        
        # 特征处理
        out = self.freq_conv(x_enhanced)
        
        return out


class MultiScaleEdgeDetector(nn.Module):
    """多尺度边缘检测器
    
    设计：
    - 使用不同尺寸的Sobel/Laplacian算子
    - 捕获不同粗细的边缘
    - 适应不同距离的小目标
    """
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleEdgeDetector, self).__init__()
        
        # Sobel算子（3x3）
        self.sobel_3x3 = self._create_sobel_layer(in_channels, out_channels, 3)
        
        # Sobel算子（5x5）
        self.sobel_5x5 = self._create_sobel_layer(in_channels, out_channels, 5)
        
        # Laplacian算子
        self.laplacian = self._create_laplacian_layer(in_channels, out_channels)
        
        # 边缘融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _create_sobel_layer(self, in_channels, out_channels, kernel_size):
        """创建Sobel边缘检测层"""
        padding = kernel_size // 2
        
        # 可学习的Sobel算子
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                        padding=padding, bias=False)
        
        # 初始化为Sobel核
        if kernel_size == 3:
            # Sobel-X
            sobel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32)
            # Sobel-Y
            sobel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=torch.float32)
        else:  # 5x5
            sobel_x = torch.tensor([[-1, -2, 0, 2, 1],
                                   [-4, -8, 0, 8, 4],
                                   [-6, -12, 0, 12, 6],
                                   [-4, -8, 0, 8, 4],
                                   [-1, -2, 0, 2, 1]], dtype=torch.float32)
            sobel_y = sobel_x.t()
        
        # 初始化卷积核
        with torch.no_grad():
            for i in range(out_channels):
                if i % 2 == 0:
                    conv.weight[i, :, :, :] = sobel_x.unsqueeze(0).repeat(in_channels, 1, 1) / (kernel_size * kernel_size)
                else:
                    conv.weight[i, :, :, :] = sobel_y.unsqueeze(0).repeat(in_channels, 1, 1) / (kernel_size * kernel_size)
        
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _create_laplacian_layer(self, in_channels, out_channels):
        """创建Laplacian边缘检测层"""
        conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        
        # Laplacian核
        laplacian = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32)
        
        with torch.no_grad():
            for i in range(out_channels):
                conv.weight[i, :, :, :] = laplacian.unsqueeze(0).repeat(in_channels, 1, 1) / 9.0
        
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 多尺度边缘检测
        edge_3x3 = self.sobel_3x3(x)
        edge_5x5 = self.sobel_5x5(x)
        edge_lap = self.laplacian(x)
        
        # 融合
        edge_fused = self.fusion(torch.cat([edge_3x3, edge_5x5, edge_lap], dim=1))
        
        return edge_fused

        
class HighFreqSubbandExtractor(nn.Module):
    """
    高频子带特征提取器
    专门处理小波变换的高频分量 (HL, LH, HH)
    """

    def __init__(self, in_channels, out_channels):
        super(HighFreqSubbandExtractor, self).__init__()

        # 分别处理三个高频子带
        self.hl_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

        self.lh_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

        self.hh_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels//4 * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 边缘增强卷积
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels//8),
            nn.Conv2d(out_channels, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_coeffs):
        """
        high_coeffs: shape [B, C, 3, H, W] 包含HL, LH, HH三个子带
        """
        # 分离三个高频子带
        hl = high_coeffs[:, :, 0, :, :]  # 水平边缘
        lh = high_coeffs[:, :, 1, :, :]  # 垂直边缘
        hh = high_coeffs[:, :, 2, :, :]  # 对角边缘

        # 分别处理
        hl_feat = self.hl_conv(hl)
        lh_feat = self.lh_conv(lh)
        hh_feat = self.hh_conv(hh)

        # 融合高频特征
        fused = torch.cat([hl_feat, lh_feat, hh_feat], dim=1)
        fused_feat = self.fusion_conv(fused)

        # 边缘增强
        edge_feat = self.edge_enhance(fused_feat)

        return fused_feat + F.interpolate(edge_feat, size=fused_feat.shape[-2:], mode='bilinear', align_corners=False)


class EdgeAttentionModule(nn.Module):
    """
    边缘注意力模块
    增强边缘特征，抑制背景噪声
    """

    def __init__(self, channels):
        super(EdgeAttentionModule, self).__init__()

        # Sobel边缘检测核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(channels, 1, 1, 1))

        # 注意力计算
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat, freq_feat):
        """
        spatial_feat: 空域特征
        freq_feat: 频域高频特征
        """
        # Sobel边缘检测
        edge_x = F.conv2d(spatial_feat, self.sobel_x, padding=1, groups=spatial_feat.size(1))
        edge_y = F.conv2d(spatial_feat, self.sobel_y, padding=1, groups=spatial_feat.size(1))
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)

        # 将频域特征上采样到空域特征尺寸
        freq_feat_upsampled = F.interpolate(freq_feat, size=spatial_feat.shape[-2:],
                                          mode='bilinear', align_corners=False)

        # 融合边缘信息和频域信息
        combined = torch.cat([spatial_feat, edge_magnitude, freq_feat_upsampled], dim=1)
        attention_weights = self.attention_conv(combined)

        # 应用注意力权重
        enhanced_feat = spatial_feat * attention_weights

        return enhanced_feat


class MultiscaleFreqFusion(nn.Module):
    """
    多尺度频域特征融合模块
    """

    def __init__(self, base_channels, num_levels):
        super(MultiscaleFreqFusion, self).__init__()

        self.num_levels = num_levels

        # 各尺度特征调整
        self.level_adjusters = nn.ModuleList()
        for i in range(num_levels):
            in_channels = base_channels * (2 ** (i + 1))  # 高频特征通道数
            self.level_adjusters.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, base_channels, 1),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * (num_levels + 1), base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_freq_features, low_freq_features, enhanced_feat):
        """
        high_freq_features: 各尺度高频特征列表
        low_freq_features: 各尺度低频特征列表
        enhanced_feat: 边缘增强特征
        """
        target_size = enhanced_feat.shape[-2:]

        # 调整各尺度特征到统一尺寸
        adjusted_features = [enhanced_feat]

        for i, high_freq in enumerate(high_freq_features):
            # 调整通道数
            adjusted = self.level_adjusters[i](high_freq)
            # 调整空间尺寸
            adjusted = F.interpolate(adjusted, size=target_size, mode='bilinear', align_corners=False)
            adjusted_features.append(adjusted)

        # 融合所有特征
        fused = torch.cat(adjusted_features, dim=1)
        output = self.fusion_conv(fused)

        return output


if __name__ == "__main__":
    # 测试频域特征提取器
    model = FrequencyFeatureExtractor(in_channels=1, base_channels=64, num_levels=3)

    # 模拟红外图像输入
    input_tensor = torch.randn(2, 1, 256, 256)

    output = model(input_tensor)

    print("High frequency features shapes:")
    for i, feat in enumerate(output['high_freq_features']):
        print(f"  Level {i}: {feat.shape}")

    print(f"Enhanced feature shape: {output['enhanced_feat'].shape}")
    print(f"Fused feature shape: {output['fused_feat'].shape}")