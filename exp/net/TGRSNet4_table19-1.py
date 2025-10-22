import torch
import torch.nn as nn
import torch.nn.functional as F
from model.block.CDConv import CDC_conv
from model.block.dual_channel_offsetor import DualChannelLocalSimGuidedSampler



class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 使用更高效的卷积层
        hidden_dim = max(in_planes // ratio, 8)  # 确保最小维度
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class EdgeBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EdgeBottleNeck, self).__init__()

        self.conv1 = CDC_conv(in_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = CDC_conv(out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FusionBlock(nn.Module):

    def __init__(self, input_channel):
        super(FusionBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channel, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU()
        )

        self.channel1 = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=1,
                                                groups=min(input_channel, 16), bias=False))
        self.channel2 = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=1,
                                                groups=min(input_channel, 16), bias=False))
        self.spatial1 = nn.Sequential(nn.Conv2d(input_channel, 1, kernel_size=1,
                                                bias=False))

        self.relu = nn.ReLU()
        self.logit = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1),
                                   nn.GroupNorm(16, input_channel))

    def forward(self, spatial, edge):
        # b, c, h, w = spatial.size()
        # [b, c*2, h, w]
        cat = torch.cat([spatial, edge], dim=1)
        f = self.conv(cat)

        amaxp = F.adaptive_max_pool2d(f, output_size=1)
        aavgp = F.adaptive_avg_pool2d(f, output_size=1)

        channel_weight = self.channel1(self.relu(amaxp)) + self.channel1(self.relu(aavgp))
        f1 = f * torch.sigmoid(self.channel2(channel_weight))
        f2 = f * torch.sigmoid(self.spatial1(spatial))

        logit = self.logit(f1 + f2)

        return logit


class TGRS(nn.Module):
    def __init__(self, input_channels, block1=BottleNeck, block2=EdgeBottleNeck):
        super(TGRS, self).__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # 图像分支
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block1)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block1, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block1, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block1, param_blocks[2])

        # 边缘分支
        self.edge_conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.edge_encoder_0 = self._make_edge_layer(param_channels[0], param_channels[0], block2)
        self.edge_encoder_1 = self._make_edge_layer(param_channels[0], param_channels[1], block2, param_blocks[0])
        self.edge_encoder_2 = self._make_edge_layer(param_channels[1], param_channels[2], block2, param_blocks[1])
        self.edge_encoder_3 = self._make_edge_layer(param_channels[2], param_channels[3], block2, param_blocks[2])
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block1, param_blocks[3])

        # 解码器
        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block1,
                                          param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block1,
                                          param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block1,
                                          param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block1)

        # 边缘和空间编码融合模块
        self.fusion1 = FusionBlock(param_channels[0])
        self.fusion2 = FusionBlock(param_channels[1])
        self.fusion3 = FusionBlock(param_channels[2])
        self.fusion4 = FusionBlock(param_channels[3])

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)
        # 交叉特征融合模块

        self.pwconv0 = nn.Sequential(
            nn.Conv2d(param_channels[0] + param_channels[1], param_channels[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(param_channels[0]),
            nn.ReLU()
        )

        self.pwconv1 = nn.Sequential(
            nn.Conv2d(param_channels[0] + param_channels[1] + param_channels[2], param_channels[1], 1, 1, 0,
                      bias=False),
            nn.BatchNorm2d(param_channels[1]),
            nn.ReLU()
        )

        self.pwconv2 = nn.Sequential(
            nn.Conv2d(param_channels[1] + param_channels[2] + param_channels[3], param_channels[2], 1, 1, 0,
                      bias=False),
            nn.BatchNorm2d(param_channels[2]),
            nn.ReLU()
        )

        self.pwconv3 = nn.Sequential(
            nn.Conv2d(param_channels[2] + param_channels[3], param_channels[3], 1, 1, 0, bias=False),
            nn.BatchNorm2d(param_channels[3]),
            nn.ReLU()
        )
        # ===== 添加LocalSimGuidedSampler模块用于解码器上采样 =====
        # 为每个解码器层创建一个sampler
        # groups参数可以根据通道数调整，这里使用channels//16作为groups

        self.sampler_d3 = DualChannelLocalSimGuidedSampler(
            hf_channels=param_channels[3],
            lf_channels=param_channels[4],
            scale=2,
            groups=param_channels[3] // 32,
            kernel_size=1
        )
        self.sampler_d2 = DualChannelLocalSimGuidedSampler(
            hf_channels=param_channels[2],
            lf_channels=param_channels[3],
            scale=2,
            groups=param_channels[2] // 16,
            kernel_size=1
        )
        self.sampler_d1 = DualChannelLocalSimGuidedSampler(
            hf_channels=param_channels[1],
            lf_channels=param_channels[2],
            scale=2,
            groups=param_channels[1] // 8,
            kernel_size=1
        )
        self.sampler_d0 = DualChannelLocalSimGuidedSampler(
            hf_channels=param_channels[0],
            lf_channels=param_channels[1],
            scale=2,
            groups=param_channels[0] // 4,
            kernel_size=1
        )

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def _make_edge_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, edge, multiscale_loss):
        # 图像分支
        # [b, 16, 256, 256]
        x_e0 = self.encoder_0(self.conv_init(x))
        # [b, 32, 128, 128]
        x_e1 = self.encoder_1(self.pool(x_e0))
        # [b, 64, 64, 64]
        x_e2 = self.encoder_2(self.pool(x_e1))
        # [b, 128, 32, 32]
        x_e3 = self.encoder_3(self.pool(x_e2))

        # 边缘分支
        e0 = self.edge_encoder_0(self.conv_init(edge))
        e1 = self.edge_encoder_1(self.pool(e0))
        e2 = self.edge_encoder_2(self.pool(e1))
        e3 = self.edge_encoder_3(self.pool(e2))

        # 简单融合
        x_e0 = self.fusion1(x_e0, e0)
        x_e1 = self.fusion2(x_e1, e1)
        x_e2 = self.fusion3(x_e2, e2)
        x_e3 = self.fusion4(x_e3, e3)

        # 交叉多尺度特征融合
        x0 = self.pwconv0(torch.cat([x_e0, self.up(x_e1)], dim=1))
        x1 = self.pwconv1(torch.cat([x_e1, self.downsample(x_e0), self.up(x_e2)], dim=1))
        x2 = self.pwconv2(torch.cat([x_e2, self.downsample(x_e1), self.up(x_e3)], dim=1))
        x3 = self.pwconv3(torch.cat([x_e3, self.downsample(x_e2)], dim=1))

        # 中间层
        x_m = self.middle_layer(self.pool(x_e3))

        # [4, 256, 16, 16]
        # 解码器引导的特征上采样层
        x_m_upsampled = self.sampler_d3(hf=x3, lf=x_m, feat2sample=x_m)
        x_d3 = self.decoder_3(torch.cat([x3, x_m_upsampled], 1))

        # decoder_2: 使用x2(hf)和x_d3(lf)来指导x_d3的上采样
        # [b, 128, 32, 32] -> [b, 128, 64, 64]
        x_d3_upsampled = self.sampler_d2(hf=x2, lf=x_d3, feat2sample=x_d3)
        x_d2 = self.decoder_2(torch.cat([x2, x_d3_upsampled], 1))

        # decoder_1: 使用x1(hf)和x_d2(lf)来指导x_d2的上采样
        # [b, 64, 64, 64] -> [b, 64, 128, 128]
        x_d2_upsampled = self.sampler_d1(hf=x1, lf=x_d2, feat2sample=x_d2)
        x_d1 = self.decoder_1(torch.cat([x1, x_d2_upsampled], 1))

        # decoder_0: 使用x0(hf)和x_d1(lf)来指导x_d1的上采样
        # [b, 32, 128, 128] -> [b, 32, 256, 256]
        x_d1_upsampled = self.sampler_d0(hf=x0, lf=x_d1, feat2sample=x_d1)
        x_d0 = self.decoder_0(torch.cat([x0, x_d1_upsampled], 1))

        if multiscale_loss:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            return [mask0, mask1, mask2, mask3], output

        else:
            output = self.output_0(x_d0)
            return [], output


if __name__ == '__main__':
    model = TGRS(input_channels=3)

    x = torch.randn(2, 3, 256, 256)
    edge = torch.randn(2, 3, 256, 256)

    # 测试warm_flag=True
    masks, output = model(x, edge,True)
    print(f"Masks: {len(masks)}, Output shape: {output.shape}")

    # 测试warm_flag=False
    _, output = model(x, edge, False)
    print(f"Output shape: {output.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")