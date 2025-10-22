import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, ic=3, oc=16):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=3//2,bias=False),
                                  nn.BatchNorm2d(oc),
                                  nn.ReLU(True))

    def forward(self, x):
        x = self.conv(x)

        return x


class CDC_conv(nn.Module):
    # 这里的theta可以设计成一个可训练的参数
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 theta=0.6,
                 padding_mode='zeros'):
        super(CDC_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              dilation=dilation,
                              bias=bias,
                              padding_mode=padding_mode)
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        if (self.theta - 0.0) < 1e-6:
            return norm_out
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            diff_out = F.conv2d(input=x,
                                weight=kernel_diff,
                                bias=self.conv.bias,
                                stride=self.conv.stride,
                                dilation=1,
                                padding=0)

            out = norm_out - self.theta * diff_out
            return out


class Adaptive_CDC_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, initial_theta=0.7):
        super(Adaptive_CDC_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding)
        # 让theta成为可学习参数
        self.theta = nn.Parameter(torch.tensor(initial_theta))

        # 添加attention机制来自适应调整theta
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm_out = self.conv(x)

        # 自适应theta调整
        att_weight = self.attention(x)
        adaptive_theta = self.theta * att_weight

        if torch.abs(adaptive_theta).max() < 1e-6:
            return norm_out

        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        diff_out = F.conv2d(x, kernel_diff, self.conv.bias,
                            self.conv.stride, dilation=1, padding=0)

        out = norm_out - adaptive_theta * diff_out
        return out


if __name__ == '__main__':
    ...