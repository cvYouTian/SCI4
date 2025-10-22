import torch
import torch.nn as nn
from model.block.sim_offsetor import DualChannelLocalSimGuidedSampler
from model.block.Fusion import FusionBlock
from model.SpaBone import AttBottleNeck
from model.EdgeBone import EdgeBottleNeck


class TGRS(nn.Module):
    def __init__(self, input_channels, block1=AttBottleNeck, block2=EdgeBottleNeck):
        super(TGRS, self).__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # 图像分支
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        # 编码器初始化
        self.encoders = nn.ModuleList()
        for i in range(4):
            if i == 0:
                # 第一层：输入输出通道相同，无block_num参数
                encoder = self._make_layer(param_channels[0], param_channels[0], block1)
            else:
                # 后续层：使用不同的通道数和block_num
                encoder = self._make_layer(param_channels[i-1], param_channels[i], block1, param_blocks[i-1])
            self.encoders.append(encoder)

        # 边缘分支
        self.edge_conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        # 边缘编码器初始化
        self.edge_encoders = nn.ModuleList()
        for i in range(4):
            if i == 0:
                # 第一层：输入输出通道相同，无block_num参数
                edge_encoder = self._make_layer(param_channels[0], param_channels[0], block2)
            else:
                # 后续层：使用不同的通道数和block_num
                edge_encoder = self._make_layer(param_channels[i-1], param_channels[i], block2, param_blocks[i-1])
            self.edge_encoders.append(edge_encoder)



        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block1, param_blocks[3])

        # 边缘和空间编码融合模块
        self.fusion_encoders = nn.ModuleList()
        for i in range(4):
            self.fusion_encoders.append(FusionBlock(param_channels[i]))

        # 解码器
        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block1,
                                          param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block1,
                                          param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block1,
                                          param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block1)

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

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

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
    

    def forward(self, x, edge, warm_flag):
        # 图像分支
        # [b, 16, 256, 256]
        x_e0 = self.encoders[0](self.conv_init(x))
        # [b, 32, 128, 128]
        x_e1 = self.encoders[1](self.pool(x_e0))
        # [b, 64, 64, 64]
        x_e2 = self.encoders[2](self.pool(x_e1))
        # [b, 128, 32, 32]
        x_e3 = self.encoders[3](self.pool(x_e2))

        # 边缘分支
        # e0 = self.edge_encoders[0](self.edge_conv_init(edge))
        # e1 = self.edge_encoders[1](self.pool(e0))
        # e2 = self.edge_encoders[2](self.pool(e1))
        # e3 = self.edge_encoders[3](self.pool(e2))

        # 两分支的融合
        # f1 = self.fusion_encoders[0](x_e0, e0)
        # f2 = self.fusion_encoders[1](x_e1, e1)
        # f3 = self.fusion_encoders[2](x_e2, e2)
        # f4 = self.fusion_encoders[3](x_e3, e3)

        f1 = x_e0
        f2 = x_e1
        f3 = x_e2
        f4 = x_e3

        # [4, 256, 16, 16]
        x_m = self.middle_layer(self.pool(f4))

        # [4, 256, 16, 16]
        # 解码器引导的特征上采样层
        x_m_upsampled = self.sampler_d3(hf=f4, lf=x_m)
        x_d3 = self.decoder_3(torch.cat([f4, x_m_upsampled], 1))

        # decoder_2: 使用x2(hf)和x_d3(lf)来指导x_d3的上采样
        # [b, 128, 32, 32] -> [b, 128, 64, 64]
        x_d3_upsampled = self.sampler_d2(hf=f3, lf=x_d3)
        x_d2 = self.decoder_2(torch.cat([f3, x_d3_upsampled], 1))

        # decoder_1: 使用x1(hf)和x_d2(lf)来指导x_d2的上采样
        # [b, 64, 64, 64] -> [b, 64, 128, 128]
        x_d2_upsampled = self.sampler_d1(hf=f2, lf=x_d2)
        x_d1 = self.decoder_1(torch.cat([f2, x_d2_upsampled], 1))

        # decoder_0: 使用x0(hf)和x_d1(lf)来指导x_d1的上采样
        # [b, 32, 128, 128] -> [b, 32, 256, 256]
        x_d1_upsampled = self.sampler_d0(hf=f1, lf=x_d1)
        x_d0 = self.decoder_0(torch.cat([f1, x_d1_upsampled], 1))

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
    print(sum(p.numel() for p in TGRS(3).parameters()))