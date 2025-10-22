import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np


class LocalSimGuidedSampler(nn.Module):
    """
    resample for offset feature
    """
    def __init__(self,
                 in_channels,
                 scale=2,
                 style="lp",
                 groups=4,
                 use_direct_scale=True,
                 kernel_size=1,
                 local_window=3,
                 sim_type="cos",
                 norm=True,
                 direction_feat="sim_concat"):
        super(LocalSimGuidedSampler).__init__()
        assert scale == 2
        assert style == "lp"

        self.scale = scale
        self.style = style
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        if style == "pl":
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        if self.direction_feat == "sim":
            self.offset = nn.Conv2d(local_window ** 2 - 1,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2)
        elif self.direction_feat == "sim_concat":
            self.offset = nn.Conv2d(in_channels + local_window ** 2 -1,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2)
        else:
            raise NotImplementedError


    def forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)

        hr_x = self.norm_hr()
        lr_x = self.norm_lr()


class ResampleFusion(nn.Module):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 feature_resample_group=4):
        super(ResampleFusion, self).__init__()
        self.dysampler = LocalSimGuidedSampler()
