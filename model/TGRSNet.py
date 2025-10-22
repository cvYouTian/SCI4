import torch
import torch.nn as nn
from model.block.upsampler import ReUpSampler
from model.SpaBone import AttBottleNeck
from model.block.downsampler import ReDownSample


class TGRS(nn.Module):
    def __init__(self, input_channels, block1=AttBottleNeck):
        super(TGRS, self).__init__()
        param_channels = [16, 32, 64, 128, 256]

        self.hf_down1 = ReDownSample(param_channels[0], scale=2)
        self.hf_down2 = ReDownSample(param_channels[1], scale=2)
        self.hf_down3 = ReDownSample(param_channels[2], scale=2)
        self.hf_down4 = ReDownSample(param_channels[3], scale=2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # 图像分支
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        # 编码器初始化
        self.encoders = nn.ModuleList()
        self.encoders.append(self._make_layer(param_channels[0], param_channels[0], block1))
        self.encoders.append(self._make_layer(param_channels[0], param_channels[1], block1, 2))
        self.encoders.append(self._make_layer(param_channels[1], param_channels[2], block1, 2))
        self.encoders.append(self._make_layer(param_channels[2], param_channels[3], block1, 2))

        # 最后一层编码器
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block1, 2)

        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(4):
            block_num = 1 if i == 0 else 2
            self.decoder.append(
                self._make_layer(param_channels[i] + param_channels[i + 1], param_channels[i], block1, block_num))

        # 特征重采样的上采样网路
        self.sampler = nn.ModuleList()
        for i in range(4):
            self.sampler.append(ReUpSampler(hf_channels=param_channels[i],
                                                                 lf_channels=param_channels[i + 1],
                                                                 scale=2,
                                                                 groups=param_channels[i] // 2 ** (2 + i),
                                                                 kernel_size=1))
        # 多尺度处理
        self.output = nn.ModuleList()
        for i in range(4):
            self.output.append(nn.Conv2d(param_channels[i], 1, 1))
        # 多尺度融合卷积
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        if block:
            layer.append(block(in_channels, out_channels))
            for _ in range(block_num - 1):
                layer.append(block(out_channels, out_channels))
        else:
            raise ValueError(f"Invalid block type: {block}")

        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        # [b, 16, 256, 256]
        f1 = self.encoders[0](self.conv_init(x))
        # [b, 16, 128, 128]
        f1_down = self.hf_down1(f1)
        # [b, 32, 128, 128]
        f2 = self.encoders[1](f1_down)
        # [b, 32, 64, 64]
        f2_down = self.hf_down2(f2)
        # [b, 64, 64, 64]
        f3 = self.encoders[2](f2_down)
        # [b, 64, 32, 32]
        f3_down = self.hf_down3(f3)
        # [b, 128, 32, 32]
        f4 = self.encoders[3](f3_down)
        # [b, 128, 16, 16]
        f4_down = self.hf_down4(f4)
        # [b, 256, 16, 16]
        x_m = self.middle_layer(f4_down)

        x_m_upsampled = self.sampler[3](hf=f4, lf=x_m)

        x_d3 = self.decoder[3](torch.cat([f4, x_m_upsampled], 1))

        # decoder_2: 使用x2(hf)和x_d3(lf)来指导x_d3的上采样
        # [b, 128, 32, 32] -> [b, 128, 64, 64]
        x_d3_upsampled = self.sampler[2](hf=f3, lf=x_d3)
        x_d2 = self.decoder[2](torch.cat([f3, x_d3_upsampled], 1))

        # decoder_1: 使用x1(hf)和x_d2(lf)来指导x_d2的上采样
        # [b, 64, 64, 64] -> [b, 64, 128, 128]
        x_d2_upsampled = self.sampler[1](hf=f2, lf=x_d2)
        x_d1 = self.decoder[1](torch.cat([f2, x_d2_upsampled], 1))

        # decoder_0: 使用x0(hf)和x_d1(lf)来指导x_d1的上采样
        # [b, 32, 128, 128] -> [b, 32, 256, 256]
        x_d1_upsampled = self.sampler[0](hf=f1, lf=x_d1)
        x_d0 = self.decoder[0](torch.cat([f1, x_d1_upsampled], 1))

        if warm_flag:
            mask0 = self.output[0](x_d0)
            mask1 = self.output[1](x_d1)
            mask2 = self.output[2](x_d2)
            mask3 = self.output[3](x_d3)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            return [mask0, mask1, mask2, mask3], output

        else:
            output = self.output[0](x_d0)
            return [], output


if __name__ == '__main__':
    print(sum(p.numel() for p in TGRS(3).parameters()))
