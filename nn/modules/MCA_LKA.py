import torch
import torch.nn as nn
import torchvision
import math

__all__ = ['DeformMCALayer', 'IntegratedModel'] # 定义模块中应该导出的类名

# 定义标准差池化层
class StdPool(nn.Module):
    def forward(self, x):
        std = torch.std(x, dim=(2, 3), keepdim=True) # 计算输入x的每个通道在空间维度上的标准差
        return std # 返回标准差

# 定义可变形多通道注意力层
class DeformMCALayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(DeformMCALayer, self).__init__()
        # 定义偏移量卷积层，用于生成可变形卷积的偏移量
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, bias=bias)
        # 定义可变形卷积层
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                                        stride=stride, padding=padding, dilation=dilation, bias=bias)
        # 自适应平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 标准差池化层
        self.std_pool = StdPool()
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        offset = self.offset_conv(x) # 生成偏移量
        x = self.deform_conv(x, offset) # 应用可变形卷积

        avg_pooled = self.avg_pool(x) # 应用平均池化
        std_pooled = self.std_pool(x) # 应用标准差池化

        attention = self.sigmoid(avg_pooled + std_pooled) # 计算注意力权重
        attention = attention.expand_as(x) # 扩展注意力权重以匹配输入x的形状
        return x * attention # 将注意力权重应用于输入x

# 定义集成模型
class IntegratedModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IntegratedModel, self).__init__()
        # 第一层为可变形多通道注意力层
        self.layer1 = DeformMCALayer(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1,
                                     groups=in_channels)
        # 第二层为标准的2D卷积层
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        # 激活函数
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x) # 通过第一层
        x = self.layer2(x) # 通过第二层
        x = self.activation(x) # 应用激活函数
        return x # 返回输出









