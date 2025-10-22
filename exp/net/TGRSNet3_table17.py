import torch
import torch.nn as nn
from model.block.CDConv import CDCConv
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
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

        self.conv1 = CDCConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CDCConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
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
    """
    table15
    """

    def __init__(self, ic):
        super(FusionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * ic, ic, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ic),
            nn.ReLU()
        )

        self.channel1 = nn.Sequential(nn.Conv2d(ic, ic, kernel_size=1, groups=16, bias=False))
        self.channel2 = nn.Sequential(nn.Conv2d(ic, ic, kernel_size=1, groups=16, bias=False))
        self.spatial1 = nn.Sequential(nn.Conv2d(ic, 1, kernel_size=1, bias=False))

        self.relu = nn.ReLU()
        self.logit = nn.Sequential(nn.Conv2d(ic, ic, kernel_size=3, padding=1),
                                   nn.GroupNorm(16, ic)
                                   )

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

        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block1,
                                          param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block1,
                                          param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block1,
                                          param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block1)

        self.fusion1 = FusionBlock(param_channels[0])
        self.fusion2 = FusionBlock(param_channels[1])
        self.fusion3 = FusionBlock(param_channels[2])
        self.fusion4 = FusionBlock(param_channels[3])

        # self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

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

    def forward(self, x, edge, warm_flag):
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

        # simple fusion
        x_e0 = self.fusion1(x_e0, e0)
        x_e1 = self.fusion2(x_e1, e1)
        x_e2 = self.fusion3(x_e2, e2)
        x_e3 = self.fusion4(x_e3, e3)

        # get the output of encoder
        x0 = self.pwconv0(torch.cat([x_e0, self.up(x_e1)], dim=1))
        x1 = self.pwconv1(torch.cat([x_e1, self.downsample(x_e0), self.up(x_e2)], dim=1))
        x2 = self.pwconv2(torch.cat([x_e2, self.downsample(x_e1), self.up(x_e3)], dim=1))
        x3 = self.pwconv3(torch.cat([x_e3, self.downsample(x_e2)], dim=1))

        x_m = self.middle_layer(self.pool(x_e3))
        # [4, 256, 16, 16]

        x_d3 = self.decoder_3(torch.cat([x3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x0, self.up(x_d1)], 1))

        if warm_flag:
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
    fusion_block = FusionBlock(ic=64)
    print(fusion_block)

    # 2. 生成随机输入
    spatial = torch.randn(2, 64, 128, 128)  # [B, C, H, W]
    edge = torch.randn(2, 64, 128, 128)

    # 3. 前向计算
    output = fusion_block(spatial, edge)

    # 4. 打印形状
    print("输入 spatial shape:", spatial.shape)
    print("输入 edge shape:   ", edge.shape)
    print("输出 shape:       ", output.shape)
