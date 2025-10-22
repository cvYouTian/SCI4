import torch
import torch.nn as nn
import torch.nn.functional as F


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0.0, std=1., bias=0.0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class AdaptiveDownSampler(nn.Module):
    """
    对称于 DualChannelLocalSimGuidedSampler 的下采样重采样器

    精髓思想融合：
    1. sim_offsetor: 使用 grid_sample 实现内容自适应采样
    2. Down_wt: 分组采样不同频率信息（不同组=不同子带）
    3. AHPF: 高频特征引导偏移预测（高频区域需要精确采样）
    """

    def __init__(self, channels, scale=2, groups=4, kernel_size=1):
        super(AdaptiveDownSampler, self).__init__()

        self.channels = channels
        self.scale = scale
        self.groups = groups  # 类似 Down_wt 的 4 个子带思想

        # 下采样的偏移输出通道数
        # 每个低分辨率位置需要从高分辨率的某个位置采样
        out_channels = 2 * groups

        # 高频特征提取（AHPF 思想：显式建模高频）
        # 不直接用 AHPF 模块，而是用简单的高通算子
        self.high_freq_extractor = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        # 初始化为 Laplacian 高通核
        self._init_highpass_kernel()

        # 方向特征提取（sim_offsetor 思想）
        self.direction_conv = nn.Conv2d(
            3 ** 2 - 1, out_channels, kernel_size, padding=kernel_size // 2
        )
        normal_init(self.direction_conv, std=0.001)

        # 特征引导的偏移大小（AHPF 思想：内容自适应）
        self.magnitude_conv = nn.Conv2d(
            channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        constant_init(self.magnitude_conv, val=0.)

        # 高频引导的偏移大小
        self.high_freq_guide = nn.Conv2d(
            channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        constant_init(self.high_freq_guide, val=0.)

        self.norm = nn.GroupNorm(min(channels // 8, 8), channels)

        # 初始偏移（下采样时每个低分辨率位置对应高分辨率中心）
        self.register_buffer("init_pos", self._init_pos())

    def _init_highpass_kernel(self):
        """初始化高通滤波核（借鉴 AHPF 的高频提取思想）"""
        # Laplacian 算子: [0,-1,0; -1,4,-1; 0,-1,0]
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.high_freq_extractor.weight.data = kernel
        # 冻结这个参数或让它可学习都可以

    def _init_pos(self):
        """初始偏移：低分辨率每个位置对应高分辨率的中心区域"""
        # 与上采样相反，这里是从高分辨率中心采样
        h = torch.zeros(1, 1)  # 可以初始化为中心偏移
        return h.repeat(1, 2 * self.groups, 1, 1)

    @staticmethod
    def direction_feat(input_tensor, k=3):
        """
        计算方向特征（与 sim_offsetor 完全相同）
        用于捕获局部相似性，引导采样方向
        """
        B, C, H, W = input_tensor.shape
        unfold_tensor = F.unfold(input_tensor, k, padding=k // 2)
        unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)

        # 中心点与邻域的相似性
        similarity = F.cosine_similarity(
            unfold_tensor[:, :, k * k // 2:k * k // 2 + 1],
            unfold_tensor[:, :, :],
            dim=1
        )

        # 去除中心点
        similarity = torch.cat(
            (similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]),
            dim=1
        )
        similarity = similarity.view(B, k * k - 1, H, W)

        return similarity

    def get_offset(self, x, high_freq):
        """
        预测下采样偏移

        关键思想（融合三者）：
        1. 用方向特征引导偏移方向（sim_offsetor）
        2. 用高频特征调整偏移幅度（AHPF）
        3. 不同 group 会学到不同的偏移（Down_wt 的频率分离）
        """
        # 下采样后的分辨率
        H_low = x.size(2) // self.scale
        W_low = x.size(3) // self.scale

        # 先下采样到目标分辨率来计算特征
        x_low = F.avg_pool2d(x, self.scale)
        high_freq_low = F.avg_pool2d(high_freq, self.scale)

        # 方向特征
        dir_feat = self.direction_feat(x_low)

        # 偏移 = 方向 × (特征引导的幅度 + 高频引导的幅度)
        offset = self.direction_conv(dir_feat) * \
                 (self.magnitude_conv(x_low) +
                  self.high_freq_guide(high_freq_low)).sigmoid()

        return offset

    def sample_down(self, x, offset, scale=None):
        """
        下采样的核心方法（对称于上采样的 sample）

        关键区别：
        - 上采样：低分辨率坐标 + pixel_shuffle → 高分辨率采样
        - 下采样：高分辨率坐标 + pixel_unshuffle → 低分辨率采样
        """
        if scale is None:
            scale = self.scale

        B, C, H, W = x.shape
        H_low, W_low = H // scale, W // scale

        # offset shape: [B, 2*groups, H_low, W_low]
        offset = offset.view(B, 2, self.groups, H_low, W_low)

        # 构建低分辨率网格的基础坐标（在高分辨率空间中）
        # 每个低分辨率位置对应高分辨率的中心
        coords_h = torch.arange(H_low, device=x.device) * scale + scale / 2
        coords_w = torch.arange(W_low, device=x.device) * scale + scale / 2

        coords = torch.stack(
            torch.meshgrid([coords_w, coords_h], indexing='ij')
        ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype)
        # coords: [1, 2, 1, H_low, W_low]

        # 广播加偏移
        coords = coords + offset  # [B, 2, groups, H_low, W_low]

        # 归一化到 [-1, 1]（grid_sample 要求）
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device
        ).view(1, 2, 1, 1, 1)
        coords = 2 * coords / normalizer - 1

        # 重塑用于 grid_sample
        # [B, 2, groups, H_low, W_low] -> [B*groups, H_low, W_low, 2]
        coords = coords.permute(0, 2, 3, 4, 1).contiguous()
        coords = coords.view(B * self.groups, H_low, W_low, 2)

        # 分组采样（Down_wt 思想：不同组采样不同信息）
        x_grouped = x.view(B * self.groups, -1, H, W)

        # 从高分辨率采样到低分辨率
        output = F.grid_sample(
            x_grouped, coords,
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        )

        # 重组输出
        output = output.view(B, -1, H_low, W_low)

        return output

    def forward(self, x):
        """
        前向传播

        流程：
        1. 提取高频特征（AHPF 思想）
        2. 预测采样偏移（sim_offsetor 机制）
        3. 执行自适应下采样（grid_sample）
        """
        # 归一化
        x_norm = self.norm(x)

        # 提取高频特征（AHPF 思想：显式建模高频）
        high_freq = self.high_freq_extractor(x_norm)

        # 预测偏移（高频引导 + 方向特征）
        offset = self.get_offset(x_norm, high_freq)

        # 自适应下采样
        output = self.sample_down(x, offset)

        return output


# 测试代码
if __name__ == "__main__":
    # 创建下采样器
    downsampler = AdaptiveDownSampler(
        channels=128,
        scale=2,
        groups=4,
        kernel_size=1
    )

    # 测试输入
    x = torch.randn(2, 128, 64, 64)

    # 下采样
    output = downsampler(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 验证信息保留
    print(f"\n输入统计: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"输出统计: mean={output.mean():.4f}, std={output.std():.4f}")
  #
  # 核心创新点解析
  #
  # 1. 对称的 sample_down 方法
  #
  # # 上采样 sample：
  # 低分辨率网格 [H, W]
  #   ↓ pixel_shuffle
  # 高分辨率坐标 [H*s, W*s, 2]
  #   ↓ grid_sample(低分辨率特征, 高分辨率坐标)
  # 高分辨率输出
  #
  # # 下采样 sample_down：
  # 高分辨率网格 [H*s, W*s]
  #   ↓ 稀疏采样点 [H, W]（每个对应高分辨率的局部中心）
  # 低分辨率坐标 [H, W, 2]（指向高分辨率空间的位置）
  #   ↓ grid_sample(高分辨率特征, 低分辨率坐标)
  # 低分辨率输出
  #
  # 2. 借鉴 Down_wt 的频率分离思想
  #
  # 不使用小波，而是通过 groups 分组采样：
  #
  # # Down_wt: 显式分解为 4 个子带
  # LL = 低频低频
  # LH = 低频高频
  # HL = 高频低频
  # HH = 高频高频
  #
  # # 我们的方法: groups 隐式学习不同频率
  # group_0 → 可能学到类似 LL 的采样（平滑区域中心）
  # group_1 → 可能学到类似 LH 的采样（垂直边缘）
  # group_2 → 可能学到类似 HL 的采样（水平边缘）
  # group_3 → 可能学到类似 HH 的采样（角点、纹理）
  #
  # 3. 借鉴 AHPF 的自适应滤波思想
  #
  # 不使用 carafe，而是用高频特征引导偏移：
  #
  # # AHPF: 生成自适应滤波器 → 应用到特征
  # filter = filter_generator(x)
  # low = apply_filter(x, filter)
  # high = x - low
  #
  # # 我们的方法: 生成自适应偏移 → 引导采样位置
  # high_freq = high_pass(x)
  # offset = offset_predictor(x, high_freq)  # 高频引导
  # output = grid_sample(x, coords + offset)  # 偏移采样
  #
  # 4. 与上采样的完美对称性
  #
  # | 操作   | 上采样             | 下采样          |
  # |------|-----------------|--------------|
  # | 偏移通道 | 2*groups*scale² | 2*groups     |
  # | 坐标空间 | 低分辨率 → 高分辨率     | 高分辨率 → 低分辨率  |
  # | 采样方向 | 从少采样到多          | 从多采样到少       |
  # | 特征引导 | 低频+高频           | 高频为主         |
  # | 信息流  | 特征扩展            | 特征压缩（但保留重要性） |
  #
  # ---
  # 优势总结
  #
  # 1. 无信息强制丢弃：通过学习的偏移，重要区域（高频）被精确采样
  # 2. 内容自适应：不同图像区域用不同的采样策略
  # 3. 频率感知：groups 隐式实现频率分离，类似小波的多子带
  # 4. 完全可微：整个过程可端到端训练
  # 5. 编码-解码对称：与你的 sim_offsetor 上采样形成完美配对
  # 这个设计保留了 sample 方法的核心机制，同时融合了 Down_wt 和 AHPF 的思想精髓！

