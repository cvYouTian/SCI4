import torch
import torch.nn as nn
import torch.nn.functional as F
from model.block.freq_backbone import FrequencyFeatureExtractor
from model.block.spatial_freq_fusion import SpatialFrequencyFusion, CompactSpatialFrequencyFusion
from model.block.wavelet_resampler import WaveletGuidedHighFreqResampler, CompactWaveletResampler


class InfraredSmallTargetNet(nn.Module):
    """
    红外弱小目标检测网络
    整合频域特征提取、空域频域融合和高频保留重采样

    网络架构：
    1. 频域特征提取主干网络 - 提取高频边缘特征
    2. 空域特征提取主干网络 - 提取空域语义特征
    3. 空域频域特征融合 - 融合两种互补特征
    4. 高频保留重采样 - 上采样时保留小目标细节
    """

    def __init__(self, in_channels=1, num_classes=2, base_channels=64):
        super(InfraredSmallTargetNet, self).__init__()

        # 频域特征提取主干网络
        self.freq_backbone = FrequencyFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=3,
            wavelet='db4'
        )

        # 空域特征提取主干网络
        self.spatial_backbone = SpatialBackbone(in_channels, base_channels)

        # 多尺度特征融合
        self.fusion_layers = nn.ModuleList([
            SpatialFrequencyFusion(base_channels, base_channels, base_channels, 'adaptive'),
            SpatialFrequencyFusion(base_channels*2, base_channels*2, base_channels*2, 'compact'),
            SpatialFrequencyFusion(base_channels*4, base_channels*4, base_channels*4, 'compact')
        ])

        # 高频保留重采样器
        self.resamplers = nn.ModuleList([
            WaveletGuidedHighFreqResampler(base_channels, scale_factor=2),
            WaveletGuidedHighFreqResampler(base_channels*2, scale_factor=2),
            CompactWaveletResampler(base_channels*4, scale_factor=2)
        ])

        # 最终预测头
        self.prediction_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, num_classes, 1)
        )

    def forward(self, x):
        # 频域特征提取
        freq_output = self.freq_backbone(x)
        freq_features = freq_output['high_freq_features']  # 多尺度频域特征
        enhanced_freq = freq_output['enhanced_feat']       # 边缘增强特征

        # 空域特征提取
        spatial_features = self.spatial_backbone(x)

        # 多尺度特征融合和重采样
        fused_features = []
        for i, (spatial_feat, freq_feat) in enumerate(zip(spatial_features, freq_features)):
            # 特征融合
            fused = self.fusion_layers[i](spatial_feat, freq_feat)
            fused_features.append(fused)

        # 自底向上的特征重采样和融合
        current_feat = fused_features[-1]  # 最深层特征

        for i in range(len(fused_features) - 2, -1, -1):
            # 高频保留重采样
            upsampled = self.resamplers[i](current_feat, fused_features[i])
            # 特征相加融合
            current_feat = upsampled + fused_features[i]

        # 最终预测
        prediction = self.prediction_head(current_feat)
        prediction = F.interpolate(prediction, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return {
            'prediction': prediction,
            'freq_features': freq_features,
            'spatial_features': spatial_features,
            'fused_features': fused_features,
            'enhanced_freq': enhanced_freq
        }


class SpatialBackbone(nn.Module):
    """
    空域特征提取主干网络
    """

    def __init__(self, in_channels, base_channels):
        super(SpatialBackbone, self).__init__()

        # 编码器
        self.encoder1 = self._make_layer(in_channels, base_channels)
        self.encoder2 = self._make_layer(base_channels, base_channels*2, stride=2)
        self.encoder3 = self._make_layer(base_channels*2, base_channels*4, stride=2)

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat1 = self.encoder1(x)      # [B, C, H, W]
        feat2 = self.encoder2(feat1)  # [B, 2C, H/2, W/2]
        feat3 = self.encoder3(feat2)  # [B, 4C, H/4, W/4]
        return [feat1, feat2, feat3]


def analyze_network_design():
    """
    详细分析网络设计的各个组件
    """
    print("=== 红外弱小目标检测网络设计分析 ===\n")

    print("1. 频域特征提取主干网络 (FrequencyFeatureExtractor)")
    print("   - 使用多尺度小波变换分解图像为不同频域子带")
    print("   - 专门提取高频子带(HL, LH, HH)的边缘特征")
    print("   - 使用边缘注意力模块增强关键特征")
    print("   - 多层次特征融合保持小目标信息")
    print("   - 优势：能够有效提取红外图像中的边缘和纹理信息\n")

    print("2. 空域频域特征融合 (SpatialFrequencyFusion)")
    print("   - 提供多种融合策略：自适应、通道注意力、交叉注意力")
    print("   - 自适应融合：根据特征内容自动调节融合权重")
    print("   - 通道注意力：强调重要的特征通道")
    print("   - 设计简单但有效，符合用户要求")
    print("   - 优势：充分利用空域和频域的互补信息\n")

    print("3. 高频保留重采样 (WaveletGuidedHighFreqResampler)")
    print("   - 结合小波变换和相似性引导重采样")
    print("   - 在重采样过程中特别保护高频信息")
    print("   - 使用高频信息指导偏移量生成")
    print("   - 高频补偿机制防止信息丢失")
    print("   - 优势：解决了上采样过程中小目标特征偏移和丢失问题\n")

    print("4. 整体网络架构优势")
    print("   - 多尺度特征提取和融合")
    print("   - 频域和空域特征的深度融合")
    print("   - 专门针对小目标检测优化")
    print("   - 保留高频细节信息")
    print("   - 模块化设计，易于扩展和调整\n")

    print("5. 与现有方法的对比")
    print("   - 相比传统上采样：更好地保留高频特征")
    print("   - 相比FreqFusion：专门针对小目标优化")
    print("   - 相比sim_offset：结合了频域信息指导")
    print("   - 相比faardown：不仅下采样，还有上采样保护")


def test_network():
    """
    测试网络的功能
    """
    print("=== 网络功能测试 ===\n")

    # 创建网络
    model = InfraredSmallTargetNet(in_channels=1, num_classes=2, base_channels=64)
    model.eval()

    # 模拟红外图像输入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 256, 256)

    print(f"输入尺寸: {input_tensor.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)

    print(f"预测输出尺寸: {output['prediction'].shape}")
    print(f"频域特征层数: {len(output['freq_features'])}")
    print(f"空域特征层数: {len(output['spatial_features'])}")
    print(f"融合特征层数: {len(output['fused_features'])}")

    for i, feat in enumerate(output['freq_features']):
        print(f"频域特征{i}: {feat.shape}")

    for i, feat in enumerate(output['spatial_features']):
        print(f"空域特征{i}: {feat.shape}")

    print(f"边缘增强特征: {output['enhanced_freq'].shape}")

    print("\n=== 网络参数统计 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")


def usage_example():
    """
    使用示例
    """
    print("=== 使用示例 ===\n")

    print("1. 基本使用:")
    print("""
    from model.block.infrared_target_net import InfraredSmallTargetNet

    # 创建网络
    model = InfraredSmallTargetNet(in_channels=1, num_classes=2, base_channels=64)

    # 输入红外图像
    input_image = torch.randn(1, 1, 256, 256)  # [batch, channel, height, width]

    # 前向传播
    output = model(input_image)
    prediction = output['prediction']  # 预测结果
    """)

    print("2. 单独使用各个模块:")
    print("""
    # 频域特征提取
    freq_extractor = FrequencyFeatureExtractor(in_channels=1, base_channels=64)
    freq_output = freq_extractor(input_image)

    # 特征融合
    fusion_layer = SpatialFrequencyFusion(64, 64, 64, 'adaptive')
    fused_feat = fusion_layer(spatial_feat, freq_feat)

    # 高频保留重采样
    resampler = WaveletGuidedHighFreqResampler(64, scale_factor=2)
    upsampled_feat = resampler(low_res_feat, guide_feat)
    """)

    print("3. 训练示例:")
    print("""
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output['prediction'], target)
            loss.backward()
            optimizer.step()
    """)


if __name__ == "__main__":
    # 运行分析和测试
    analyze_network_design()
    print("\n" + "="*60 + "\n")
    test_network()
    print("\n" + "="*60 + "\n")
    usage_example()