import torch
import torch.nn as nn
import torch.nn.functional as F
from model.block.faardown import DWTForward


class ReDownSample(nn.Module):
    def __init__(self, channels, scale=2, groups=4):
        super(ReDownSample, self).__init__()
        self.scale = scale
        self.groups = groups
        self.channels = channels

        self.pool = nn.MaxPool2d(scale, scale)
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')

        self.hf_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.lf_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        out_channels = 2 * groups * scale ** 2

        self.hf_direction = nn.Conv2d(channels, out_channels, 1)
        nn.init.normal_(self.hf_direction.weight, std=0.001)
        nn.init.constant_(self.hf_direction.bias, 0)

        self.lf_direction = nn.Conv2d(channels, out_channels, 1)
        nn.init.normal_(self.lf_direction.weight, std=0.001)
        nn.init.constant_(self.lf_direction.bias, 0)

        self.hf_scale = nn.Conv2d(channels, out_channels, 1)
        nn.init.constant_(self.hf_scale.weight, 0.)
        nn.init.constant_(self.hf_scale.bias, 0.)

        self.lf_scale = nn.Conv2d(channels, out_channels, 1)
        nn.init.constant_(self.lf_scale.weight, 0.)
        nn.init.constant_(self.lf_scale.bias, 0.)

        self.register_buffer("init_pos", self._init_pos())

        self.norm_hf = nn.GroupNorm(min(channels // 8, 8), channels)
        self.norm_lf = nn.GroupNorm(min(channels // 8, 8), channels)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def get_highfreq(self, hf_feat, lf_feat):
        offset = (self.hf_direction(hf_feat) + self.lf_direction(lf_feat)) * \
                 (self.hf_scale(hf_feat) + self.lf_scale(lf_feat)).sigmoid() + self.init_pos

        return offset

    def sample(self, x, offset):
        B, C, H, W = x.shape
        out_H, out_W = H // self.scale, W // self.scale

        offset = offset.view(B, 2, -1, out_H, out_W)

        coords_h = torch.arange(out_H, device=x.device, dtype=x.dtype) * self.scale + self.scale / 2 - 0.5
        coords_w = torch.arange(out_W, device=x.device, dtype=x.dtype) * self.scale + self.scale / 2 - 0.5

        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')).\
            transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)

        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(B, -1, out_H, out_W), self.scale).view(
            B, 2, -1, H, W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode="border"
        ).view(B, -1, out_H, out_W)

    def forward(self, x):
        B, C, H, W = x.shape

        x_pool = self.pool(x)

        LL, yH = self.dwt(x)
        LH = yH[0][:, :, 0, ::]
        HL = yH[0][:, :, 1, ::]
        HH = yH[0][:, :, 2, ::]

        hf_feat = self.hf_fusion(torch.cat([LH, HL, HH], dim=1))
        lf_feat = self.lf_fusion(torch.cat([LL, x_pool], dim=1))

        hf_norm = self.norm_hf(hf_feat)
        lf_norm = self.norm_lf(lf_feat)

        offset = self.get_highfreq(hf_norm, lf_norm)

        output = self.sample(x, offset)

        return output


if __name__ == '__main__':
    model = ReDownSample(channels=64, scale=2, groups=4)
    x = torch.randn(2, 64, 128, 128)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
