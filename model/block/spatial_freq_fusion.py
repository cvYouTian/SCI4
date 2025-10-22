import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFrequencyFusion(nn.Module):
    """
    空域和频域特征融合模块

    设计思路：
    1. 简单而有效的融合策略
    2. 保持空域和频域特征的互补性
    3. 使用注意力机制自适应调节融合权重
    4. 针对小目标检测优化
    """

    def __init__(self, spatial_channels, freq_channels, output_channels, fusion_type='adaptive'):
        super(SpatialFrequencyFusion, self).__init__()

        self.fusion_type = fusion_type
        self.spatial_channels = spatial_channels
        self.freq_channels = freq_channels
        self.output_channels = output_channels

        # 通道对齐
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(spatial_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.freq_adapter = nn.Sequential(
            nn.Conv2d(freq_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        if fusion_type == 'adaptive':
            # 自适应融合权重生成
            self.weight_generator = AdaptiveFusionWeights(output_channels)
        elif fusion_type == 'channel_attention':
            # 通道注意力融合
            self.channel_attention = ChannelAttentionFusion(output_channels)
        elif fusion_type == 'cross_attention':
            # 交叉注意力融合
            self.cross_attention = CrossModalAttention(output_channels)

        # 最终特征细化
        self.feature_refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, spatial_feat, freq_feat):
        """
        spatial_feat: 空域特征 [B, C_s, H, W]
        freq_feat: 频域特征 [B, C_f, H', W']
        """
        # 确保空间尺寸一致
        if freq_feat.shape[-2:] != spatial_feat.shape[-2:]:
            freq_feat = F.interpolate(freq_feat, size=spatial_feat.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # 通道对齐
        spatial_aligned = self.spatial_adapter(spatial_feat)
        freq_aligned = self.freq_adapter(freq_feat)

        # 特征融合
        if self.fusion_type == 'simple':
            # 简单相加融合
            fused = spatial_aligned + freq_aligned
        elif self.fusion_type == 'adaptive':
            # 自适应加权融合
            fused = self.weight_generator(spatial_aligned, freq_aligned)
        elif self.fusion_type == 'channel_attention':
            # 通道注意力融合
            fused = self.channel_attention(spatial_aligned, freq_aligned)
        elif self.fusion_type == 'cross_attention':
            # 交叉注意力融合
            fused = self.cross_attention(spatial_aligned, freq_aligned)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # 特征细化
        refined = self.feature_refine(fused)

        return refined


class AdaptiveFusionWeights(nn.Module):
    """
    自适应融合权重生成器
    根据特征内容自动调节空域和频域特征的融合权重
    """

    def __init__(self, channels):
        super(AdaptiveFusionWeights, self).__init__()

        # 全局池化获取全局信息
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 权重生成网络
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),  # 输出2个权重
            nn.Softmax(dim=1)
        )

        # 局部权重生成（考虑空间变化）
        self.spatial_weight = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 8, 3, padding=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, spatial_feat, freq_feat):
        B, C, H, W = spatial_feat.shape

        # 特征拼接
        combined = torch.cat([spatial_feat, freq_feat], dim=1)

        # 全局权重
        global_feat = self.global_pool(combined)
        global_weights = self.weight_net(global_feat)  # [B, 2, 1, 1]

        # 局部权重
        local_weights = self.spatial_weight(combined)  # [B, 2, H, W]

        # 结合全局和局部权重
        alpha_global = global_weights[:, 0:1, :, :]  # 空域权重
        beta_global = global_weights[:, 1:2, :, :]   # 频域权重

        alpha_local = local_weights[:, 0:1, :, :]    # 空域权重
        beta_local = local_weights[:, 1:2, :, :]     # 频域权重

        # 融合权重
        alpha = alpha_global * alpha_local
        beta = beta_global * beta_local

        # 归一化
        weights_sum = alpha + beta + 1e-8
        alpha = alpha / weights_sum
        beta = beta / weights_sum

        # 加权融合
        fused = alpha * spatial_feat + beta * freq_feat

        return fused


class ChannelAttentionFusion(nn.Module):
    """
    通道注意力融合模块
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttentionFusion, self).__init__()

        self.channels = channels
        self.reduction = reduction

        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial_feat, freq_feat):
        B, C, H, W = spatial_feat.shape

        # 空域特征的通道注意力
        spatial_avg = F.avg_pool2d(spatial_feat, (H, W))
        spatial_max = F.max_pool2d(spatial_feat, (H, W))
        spatial_att = self.sigmoid(self.mlp(spatial_avg) + self.mlp(spatial_max))

        # 频域特征的通道注意力
        freq_avg = F.avg_pool2d(freq_feat, (H, W))
        freq_max = F.max_pool2d(freq_feat, (H, W))
        freq_att = self.sigmoid(self.mlp(freq_avg) + self.mlp(freq_max))

        # 应用注意力权重
        spatial_enhanced = spatial_feat * spatial_att
        freq_enhanced = freq_feat * freq_att

        # 融合
        fused = spatial_enhanced + freq_enhanced

        return fused


class CrossModalAttention(nn.Module):
    """
    交叉模态注意力融合
    让空域和频域特征相互指导
    """

    def __init__(self, channels, num_heads=8):
        super(CrossModalAttention, self).__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # 查询、键、值投影
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # 输出投影
        self.out_proj = nn.Conv2d(channels, channels, 1)

        self.scale = self.head_dim ** -0.5

    def forward(self, spatial_feat, freq_feat):
        B, C, H, W = spatial_feat.shape

        # 空域特征作为查询，频域特征作为键值
        q = self.q_proj(spatial_feat).view(B, self.num_heads, self.head_dim, H * W)
        k = self.k_proj(freq_feat).view(B, self.num_heads, self.head_dim, H * W)
        v = self.v_proj(freq_feat).view(B, self.num_heads, self.head_dim, H * W)

        # 计算注意力
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = torch.matmul(v, attn.transpose(-2, -1))  # [B, heads, head_dim, HW]
        out = out.view(B, C, H, W)
        out = self.out_proj(out)

        # 残差连接
        spatial_attended = spatial_feat + out

        # 反向：频域特征作为查询，空域特征作为键值
        q2 = self.q_proj(freq_feat).view(B, self.num_heads, self.head_dim, H * W)
        k2 = self.k_proj(spatial_feat).view(B, self.num_heads, self.head_dim, H * W)
        v2 = self.v_proj(spatial_feat).view(B, self.num_heads, self.head_dim, H * W)

        attn2 = torch.matmul(q2.transpose(-2, -1), k2) * self.scale
        attn2 = F.softmax(attn2, dim=-1)

        out2 = torch.matmul(v2, attn2.transpose(-2, -1))
        out2 = out2.view(B, C, H, W)
        out2 = self.out_proj(out2)

        freq_attended = freq_feat + out2

        # 最终融合
        fused = spatial_attended + freq_attended

        return fused


class CompactSpatialFrequencyFusion(nn.Module):
    """
    紧凑的空域频域特征融合模块
    更简单的设计，计算效率更高
    """

    def __init__(self, spatial_channels, freq_channels, output_channels):
        super(CompactSpatialFrequencyFusion, self).__init__()

        # 通道对齐和降维
        self.spatial_proj = nn.Conv2d(spatial_channels, output_channels // 2, 1)
        self.freq_proj = nn.Conv2d(freq_channels, output_channels // 2, 1)

        # 简单的特征交互
        self.interaction = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels // 8),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(output_channels, output_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels // 4, output_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat, freq_feat):
        # 尺寸对齐
        if freq_feat.shape[-2:] != spatial_feat.shape[-2:]:
            freq_feat = F.interpolate(freq_feat, size=spatial_feat.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # 投影
        spatial_proj = self.spatial_proj(spatial_feat)
        freq_proj = self.freq_proj(freq_feat)

        # 拼接
        fused = torch.cat([spatial_proj, freq_proj], dim=1)

        # 特征交互
        interacted = self.interaction(fused)

        # 门控调制
        gate_weights = self.gate(interacted)
        output = interacted * gate_weights

        return output


if __name__ == "__main__":
    # 测试融合模块
    batch_size = 2
    spatial_feat = torch.randn(batch_size, 128, 64, 64)
    freq_feat = torch.randn(batch_size, 256, 32, 32)

    # 测试不同融合方式
    fusion_models = {
        'adaptive': SpatialFrequencyFusion(128, 256, 128, 'adaptive'),
        'channel_attention': SpatialFrequencyFusion(128, 256, 128, 'channel_attention'),
        'cross_attention': SpatialFrequencyFusion(128, 256, 128, 'cross_attention'),
        'compact': CompactSpatialFrequencyFusion(128, 256, 128)
    }

    for name, model in fusion_models.items():
        output = model(spatial_feat, freq_feat)
        print(f"{name} fusion output shape: {output.shape}")

    print("All fusion tests passed!")