import torch
import torch.nn as nn
import torch.nn.functional as F


class ReUpSampler(nn.Module):
    """
    支持hf和lf不同通道数的ReSampler
    """
    def __init__(self, hf_channels, lf_channels, scale=2, groups=4, kernel_size=1):
        super(ReUpSampler, self).__init__()
        # 这个是lf的输出通道
        out_channels = 2 * groups * scale ** 2
        self.scale = scale
        # 通过 groups 参数将通道分组，每组学习独立的采样策略，增强表达能力（后续将这个改成4个上尝试一下效果，或者一个）
        self.groups = groups

        # 定义lf特征的方向
        self.lf_conv = nn.Conv2d(3 ** 2 - 1, out_channels, kernel_size, padding=kernel_size // 2)
        nn.init.normal_(self.lf_conv.weight, 0.0, 0.001)
        nn.init.constant_(self.lf_conv.bias, 0.0)

        # 定义lf特征的大小 (使用lf_channels)
        self.direct_scale = nn.Conv2d(lf_channels, out_channels, kernel_size, padding=kernel_size // 2)
        nn.init.constant_(self.direct_scale.weight, 0)
        nn.init.constant_(self.direct_scale.bias, 0)

        # 这个是hf的输出通道
        out_channels = 2 * groups
        # 定义hf特征方向
        self.hf_conv = nn.Conv2d(3**2-1, out_channels, kernel_size, padding=kernel_size // 2)
        nn.init.normal_(self.hf_conv.weight, 0.0, 0.001)
        nn.init.constant_(self.hf_conv.bias, 0.0)

        # 定义hf特征大小 (使用hf_channels)
        self.hr_direct_scale = nn.Conv2d(hf_channels, out_channels, kernel_size, padding=kernel_size // 2)
        nn.init.constant_(self.direct_scale.weight, 0)
        nn.init.constant_(self.direct_scale.bias, 0)

        # 是否作正则化
        self.norm_hr = nn.GroupNorm(min(hf_channels // 8, 8), hf_channels)
        self.norm_lr = nn.GroupNorm(min(lf_channels // 8, 8), lf_channels)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    @staticmethod
    def direction_feat(input_tensor, k=3):
        # 计算像素中心和它的8邻域的余弦相似性
        B, C, H, W = input_tensor.shape
        # 将3x3的邻域拿出来，作为邻域信息
        # TODO: 这个方法告诉我如何去拿到周围像素点的关系。
        unfold_tensor = F.unfold(input_tensor, k, padding=k // 2)
        # 拿到邻域关系
        unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)
        # 计算中心点和相邻点的余弦相似度
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
        # 去除中心的特征点
        similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)
        # 对结果进行reshape[b, 8, h, w]
        similarity = similarity.view(B, k * k - 1, H, W)

        return similarity

    def sample(self, resample_lf, offset, scale=None):
        if scale is None:
            scale = self.scale
        B, _, H, W = offset.shape
        # 将通道分成两个组
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')).\
            transpose(1, 2).unsqueeze(1).unsqueeze(0).type(resample_lf.dtype).to(resample_lf.device)

        normalizer = torch.tensor([W, H], dtype=resample_lf.dtype, device=resample_lf.device).view(1, 2, 1, 1, 1)
        # 我将归一化后的坐标再次乘以2减一实现了的范围到（-1， 1）的映射
        coords = 2 * (coords + offset) / normalizer - 1
        # 这里是
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(resample_lf.reshape(B * self.groups, -1, resample_lf.size(-2), resample_lf.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)

    def get_offset(self, higt_f, low_f, hf_sim, lf_sim):
        # 注意此函数只计算了偏移量
        # 对应文章中的公式10
        # 得到最终的偏移方向和大小
        offset = (self.lf_conv(lf_sim) + F.pixel_unshuffle(self.hf_conv(hf_sim), self.scale)) * \
                 (self.direct_scale(low_f) + F.pixel_unshuffle(self.hr_direct_scale(higt_f), self.scale)).sigmoid() + self.init_pos

        return offset

    def forward(self, hf, lf):
        # 这里只是对输入进行归一化，没有其他的作用
        resample_f = lf
        hf = self.norm_hr(hf)
        lf = self.norm_lr(lf)
        # 这里得到两个方向的特征矩阵【B， 8, H， W】
        hf_sim = self.direction_feat(hf)
        lf_sim = self.direction_feat(lf)
        offset = self.get_offset(hf, lf, hf_sim, lf_sim)

        # 使用两个特征偏移后的方向和大小对lf进行重新的上采样
        return self.sample(resample_f, offset)


if __name__ == "__main__":
    # 测试不同通道数
    sim = ReUpSampler(hf_channels=128, lf_channels=256, scale=2, groups=4, kernel_size=1)
    lf = torch.randn(2, 256, 16, 16)  # 低分辨率特征
    hf = torch.randn(2, 128, 32, 32)  # 高分辨率特征

    result = sim(hf, lf)
    print(f"Result shape: {result.shape}")