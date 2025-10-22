import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.block.faardown import DWTForward, AFB2D, mode_to_int, prep_filt_afb2d
from model.block.sim_offsetor import DualChannelLocalSimGuidedSampler


class WaveletGuidedHighFreqResampler(nn.Module):
    """
    基于小波变换的高频特征保留重采样器

    设计思路：
    1. 结合小波变换和相似性引导重采样
    2. 专门保留和增强高频特征
    3. 防止小目标在重采样过程中的信息丢失
    4. 使用多尺度高频信息指导重采样过程
    """

    def __init__(self, in_channels, scale_factor=2, wavelet='db4', groups=4, preserve_method='enhanced'):
        super(WaveletGuidedHighFreqResampler, self).__init__()

        self.scale_factor = scale_factor
        self.groups = groups
        self.preserve_method = preserve_method

        # 小波变换
        self.dwt = DWTForward(J=1, wave=wavelet, mode='symmetric')

        # 高频特征处理器
        self.high_freq_processor = HighFreqProcessor(in_channels, groups)

        # 基于高频信息的偏移生成器
        self.offset_generator = WaveletGuidedOffsetGenerator(
            in_channels, scale_factor, groups
        )

        # 高频保留的重采样器
        self.freq_aware_sampler = FrequencyAwareResampler(
            in_channels, scale_factor, groups
        )

        # 高频增强模块
        if preserve_method == 'enhanced':
            self.high_freq_enhancer = HighFreqEnhancer(in_channels)

        # 特征重建器
        self.feature_reconstructor = FeatureReconstructor(in_channels)

    def forward(self, feature_to_resample, guide_feature=None):
        """
        feature_to_resample: 需要重采样的特征 [B, C, H, W]
        guide_feature: 指导特征（可选）[B, C, H', W']
        """
        # 小波分解获取高频信息
        low_freq, high_freq_list = self.dwt(feature_to_resample)
        high_freq = high_freq_list[0]  # [B, C, 3, H//2, W//2]

        # 处理高频特征
        processed_high_freq = self.high_freq_processor(high_freq)

        # 生成保留高频的偏移量
        if guide_feature is not None:
            offset = self.offset_generator(feature_to_resample, guide_feature, processed_high_freq)
        else:
            offset = self.offset_generator(feature_to_resample, feature_to_resample, processed_high_freq)

        # 高频感知重采样
        resampled_feature = self.freq_aware_sampler(
            feature_to_resample, offset, low_freq, high_freq
        )

        # 高频增强
        if self.preserve_method == 'enhanced':
            resampled_feature = self.high_freq_enhancer(resampled_feature, processed_high_freq)

        # 特征重建
        final_feature = self.feature_reconstructor(resampled_feature, processed_high_freq)

        return final_feature


class HighFreqProcessor(nn.Module):
    """
    高频特征处理器
    专门处理小波变换得到的高频分量
    """

    def __init__(self, in_channels, groups):
        super(HighFreqProcessor, self).__init__()

        # 分别处理三个高频子带
        self.hl_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.lh_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.hh_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 高频特征融合
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(in_channels // 2 * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 高频增强卷积（保持边缘锐利）
        self.edge_enhance = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=groups)

    def forward(self, high_freq):
        """
        high_freq: [B, C, 3, H, W] 包含HL, LH, HH三个子带
        """
        # 分离三个高频子带
        hl = high_freq[:, :, 0, :, :]  # 水平边缘
        lh = high_freq[:, :, 1, :, :]  # 垂直边缘
        hh = high_freq[:, :, 2, :, :]  # 对角边缘

        # 分别处理
        hl_feat = self.hl_processor(hl)
        lh_feat = self.lh_processor(lh)
        hh_feat = self.hh_processor(hh)

        # 融合高频信息
        fused = torch.cat([hl_feat, lh_feat, hh_feat], dim=1)
        processed = self.freq_fusion(fused)

        # 边缘增强
        enhanced = self.edge_enhance(processed)
        output = processed + enhanced

        return output


class WaveletGuidedOffsetGenerator(nn.Module):
    """
    基于小波变换的偏移生成器
    结合高频信息生成更准确的重采样偏移
    """

    def __init__(self, in_channels, scale_factor, groups):
        super(WaveletGuidedOffsetGenerator, self).__init__()

        self.scale_factor = scale_factor
        self.groups = groups

        # 基础偏移生成器（继承自sim_offsetor的思想）
        self.base_offset_gen = DualChannelLocalSimGuidedSampler(
            hf_channels=in_channels,
            lf_channels=in_channels,
            scale=scale_factor,
            groups=groups
        )

        # 高频引导的偏移调整
        self.high_freq_offset = nn.Sequential(
            nn.Conv2d(in_channels, groups * 2, 3, padding=1),
            nn.Tanh()  # 限制偏移范围
        )

        # 偏移融合权重
        self.offset_weight = nn.Sequential(
            nn.Conv2d(in_channels * 2, groups, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feature, guide_feature, high_freq_info):
        """
        feature: 当前特征
        guide_feature: 指导特征
        high_freq_info: 高频信息
        """
        # 基础偏移（使用相似性引导）
        base_offset = self.base_offset_gen.get_offset(
            feature, guide_feature,
            self.base_offset_gen.direction_feat(feature),
            self.base_offset_gen.direction_feat(guide_feature)
        )

        # 上采样高频信息到合适尺寸
        high_freq_upsampled = F.interpolate(
            high_freq_info, size=feature.shape[-2:],
            mode='bilinear', align_corners=False
        )

        # 高频引导的偏移调整
        high_freq_offset = self.high_freq_offset(high_freq_upsampled)

        # 偏移融合权重
        combined_feat = torch.cat([feature, high_freq_upsampled], dim=1)
        fusion_weight = self.offset_weight(combined_feat)

        # 融合偏移
        # 将高频偏移reshape到与基础偏移相同的形状
        high_freq_offset_reshaped = high_freq_offset.repeat_interleave(
            self.scale_factor ** 2, dim=1
        )

        final_offset = base_offset + fusion_weight.repeat_interleave(
            2 * self.scale_factor ** 2, dim=1
        ) * high_freq_offset_reshaped * 0.1

        return final_offset


class FrequencyAwareResampler(nn.Module):
    """
    频率感知的重采样器
    在重采样过程中特别保护高频信息
    """

    def __init__(self, in_channels, scale_factor, groups):
        super(FrequencyAwareResampler, self).__init__()

        self.scale_factor = scale_factor
        self.groups = groups

        # 低频和高频分别处理的重采样
        self.low_freq_sampler = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=False
        )

        # 高频保留的网格采样
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale_factor + 1) / 2, (self.scale_factor - 1) / 2 + 1) / self.scale_factor
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample_with_offset(self, feature, offset):
        """
        使用偏移进行网格采样
        """
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H, device=feature.device) + 0.5
        coords_w = torch.arange(W, device=feature.device) + 0.5

        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(feature.dtype).to(feature.device)

        normalizer = torch.tensor([W, H], dtype=feature.dtype, device=feature.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale_factor).view(
            B, 2, -1, self.scale_factor * H, self.scale_factor * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(
            feature.reshape(B * self.groups, -1, feature.size(-2), feature.size(-1)),
            coords, mode='bilinear', align_corners=False, padding_mode="border"
        ).view(B, -1, self.scale_factor * H, self.scale_factor * W)

    def forward(self, feature, offset, low_freq, high_freq):
        """
        频率感知的重采样
        """
        # 使用偏移进行重采样
        resampled = self.sample_with_offset(feature, offset)

        # 高频信息补偿
        # 将高频信息上采样并融入结果
        high_freq_compensated = []
        for i in range(3):  # HL, LH, HH
            hf_band = high_freq[:, :, i, :, :]
            hf_upsampled = F.interpolate(
                hf_band, size=(resampled.shape[-2], resampled.shape[-1]),
                mode='bilinear', align_corners=False
            )
            high_freq_compensated.append(hf_upsampled)

        # 高频补偿加权
        total_high_freq = sum(high_freq_compensated) / 3.0
        compensated_result = resampled + 0.1 * total_high_freq

        return compensated_result


class HighFreqEnhancer(nn.Module):
    """
    高频增强模块
    进一步增强重采样后的高频特征
    """

    def __init__(self, channels):
        super(HighFreqEnhancer, self).__init__()

        # 高频细节增强
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 锐化卷积核
        sharpen_kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('sharpen_kernel', sharpen_kernel.view(1, 1, 3, 3))

    def forward(self, resampled_feature, high_freq_info):
        # 上采样高频信息
        high_freq_upsampled = F.interpolate(
            high_freq_info, size=resampled_feature.shape[-2:],
            mode='bilinear', align_corners=False
        )

        # 计算增强权重
        enhance_weight = self.detail_enhance(high_freq_upsampled)

        # 锐化操作
        sharpened = F.conv2d(
            resampled_feature,
            self.sharpen_kernel.repeat(resampled_feature.size(1), 1, 1, 1),
            padding=1, groups=resampled_feature.size(1)
        )

        # 增强融合
        enhanced = resampled_feature + enhance_weight * (sharpened - resampled_feature)

        return enhanced


class FeatureReconstructor(nn.Module):
    """
    特征重建器
    确保重采样后的特征质量
    """

    def __init__(self, channels):
        super(FeatureReconstructor, self).__init__()

        self.reconstruct = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 残差连接
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, resampled_feature, high_freq_info):
        reconstructed = self.reconstruct(resampled_feature)

        # 残差连接
        output = reconstructed + self.residual_weight * resampled_feature

        return output


class CompactWaveletResampler(nn.Module):
    """
    紧凑版本的小波引导重采样器
    更轻量级的实现
    """

    def __init__(self, in_channels, scale_factor=2, wavelet='haar'):
        super(CompactWaveletResampler, self).__init__()

        self.scale_factor = scale_factor

        # 小波变换
        self.dwt = DWTForward(J=1, wave=wavelet, mode='symmetric')

        # 简化的高频处理
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 简化的重采样
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        # 高频补偿
        self.compensation = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, feature):
        # 小波分解
        low_freq, high_freq_list = self.dwt(feature)
        high_freq = high_freq_list[0]

        # 处理高频
        hf_processed = self.high_freq_conv(high_freq.mean(dim=2))  # 平均三个高频子带

        # 上采样
        upsampled = self.upsample(feature)

        # 高频补偿
        hf_upsampled = self.upsample(hf_processed)
        compensation = self.compensation(hf_upsampled)

        # 融合
        result = upsampled + 0.1 * compensation

        return result


if __name__ == "__main__":
    # 测试重采样器
    batch_size = 2
    channels = 64
    height, width = 32, 32

    feature = torch.randn(batch_size, channels, height, width)
    guide_feature = torch.randn(batch_size, channels, height, width)

    # 测试完整版本
    resampler = WaveletGuidedHighFreqResampler(channels, scale_factor=2)
    output = resampler(feature, guide_feature)
    print(f"Full resampler output shape: {output.shape}")

    # 测试紧凑版本
    compact_resampler = CompactWaveletResampler(channels, scale_factor=2)
    compact_output = compact_resampler(feature)
    print(f"Compact resampler output shape: {compact_output.shape}")

    print("All resampler tests passed!")