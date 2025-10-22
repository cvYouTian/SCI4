"""
空域-频域特征融合模块
Spatial-Frequency Feature Fusion Block
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBlock(nn.Module):
    """
    空域和频域特征融合模块
    
    Args:
        input_channel (int): 输入特征的通道数
        
    Input:
        spatial: 空域特征 [B, C, H, W]
        edge: 频域/边缘特征 [B, C, H, W]
        
    Output:
        融合后的特征 [B, C, H, W]
    """

    def __init__(self, input_channel):
        super(FusionBlock, self).__init__()
        
        # 特征融合卷积：将拼接的特征压缩回原通道数
        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channel, input_channel, 
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU()
        )
        
        # 通道注意力：处理池化后的特征
        self.channel_process = nn.Conv2d(input_channel, input_channel, 
                                        kernel_size=1, 
                                        groups=min(input_channel, 16), 
                                        bias=False)
        
        # 通道注意力：生成权重
        self.channel_weight = nn.Conv2d(input_channel, input_channel, 
                                       kernel_size=1, 
                                       groups=min(input_channel, 16), 
                                       bias=False)
        
        # 空间注意力：生成空间权重图
        self.spatial_weight = nn.Conv2d(input_channel, 1, 
                                       kernel_size=1, 
                                       bias=False)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 最终输出层
        self.output = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1),
            nn.GroupNorm(16, input_channel)
        )

    def forward(self, spatial, edge):
        """
        前向传播
        
        Args:
            spatial: 空域特征 [B, C, H, W]
            edge: 频域/边缘特征 [B, C, H, W]
            
        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 1. 特征拼接和初步融合
        concat_features = torch.cat([spatial, edge], dim=1)  # [B, 2C, H, W]
        fused = self.conv(concat_features)  # [B, C, H, W]
        
        # 2. 通道注意力分支
        # 全局池化提取通道特征
        max_pooled = F.adaptive_max_pool2d(fused, output_size=1)  # [B, C, 1, 1]
        avg_pooled = F.adaptive_avg_pool2d(fused, output_size=1)  # [B, C, 1, 1]
        
        # 计算通道注意力权重
        channel_att = self.channel_process(self.relu(max_pooled)) + \
                     self.channel_process(self.relu(avg_pooled))  # [B, C, 1, 1]
        channel_att = torch.sigmoid(self.channel_weight(channel_att))  # [B, C, 1, 1]
        
        # 应用通道注意力
        channel_refined = fused * channel_att  # [B, C, H, W]
        
        # 3. 空间注意力分支
        # 计算空间注意力权重
        spatial_att = torch.sigmoid(self.spatial_weight(spatial))  # [B, 1, H, W]
        
        # 应用空间注意力
        spatial_refined = fused * spatial_att  # [B, C, H, W]
        
        # 4. 融合两路注意力并输出
        combined = channel_refined + spatial_refined  # [B, C, H, W]
        output = self.output(combined)  # [B, C, H, W]
        
        return output