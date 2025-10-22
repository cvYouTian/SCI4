"""
中心差分卷积 (Central Difference Convolution)
CDConv - 用于边缘和梯度特征提取

参考论文: Central Difference Convolutional Networks
"""
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """
    标准卷积块 (Conv + BN + ReLU)
    
    Args:
        in_channels (int): 输入通道数，默认3
        out_channels (int): 输出通道数，默认16
    """
    
    def __init__(self, in_channels=3, out_channels=16):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C_in, H, W]
            
        Returns:
            输出特征 [B, C_out, H, W]
        """
        return self.conv(x)


class CDCConv(nn.Module):
    """
    中心差分卷积 (Central Difference Convolution)
    
    原理：
        output = conv(x) - θ * central_diff(x)
        
    其中 central_diff 是通过对卷积核求和得到的中心差分算子，
    用于提取边缘和梯度信息。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认3
        stride (int): 步长，默认1
        padding (int): 填充，默认1
        dilation (int): 膨胀率，默认1
        bias (bool): 是否使用偏置，默认True
        theta (float): 中心差分权重，范围[0,1]，默认0.6
                      - theta=0: 退化为普通卷积
                      - theta越大，边缘增强效果越强
        padding_mode (str): 填充模式，默认'zeros'
        
    Note:
        theta 可以设置为可训练参数以自适应学习
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 theta=0.6,
                 padding_mode='zeros'):
        super(CDCConv, self).__init__()
        
        # 标准卷积层
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # 中心差分权重 (可选：改为可训练参数)
        self.theta = theta
        # self.theta = nn.Parameter(torch.tensor(theta))  # 可训练版本
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C_in, H, W]
            
        Returns:
            输出特征 [B, C_out, H, W]
        """
        # 1. 标准卷积输出
        conv_output = self.conv(x)
        
        # 2. 如果 theta ≈ 0，退化为普通卷积
        if abs(self.theta) < 1e-6:
            return conv_output
        
        # 3. 计算中心差分
        # 将卷积核在空间维度求和，得到中心差分核 [C_out, C_in, 1, 1]
        central_diff_kernel = self.conv.weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        
        # 使用中心差分核进行卷积
        diff_output = F.conv2d(
            input=x,
            weight=central_diff_kernel,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,  # 中心差分不需要padding
            dilation=1
        )
        
        # 4. 组合标准卷积和中心差分
        # output = conv(x) - θ * central_diff(x)
        output = conv_output - self.theta * diff_output
        
        return output


