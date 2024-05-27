import torch
from torch import nn
import math
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

import torch.nn as nn




class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        # 获取输入张量的尺寸信息
        b, c, _, _ = x.size()

        # 将输入张量重塑为形状为(batch_size, channels, -1)的形式，以便计算通道维度上的标准差
        std = x.view(b, c, -1).std(dim=2, keepdim=True)

        # 将计算得到的标准差张量重新调整为与输入张量相同的形状，以便进行广播操作
        std = std.reshape(b, c, 1, 1)

        # 返回标准差张量作为标准差池化操作的输出
        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """构造一个 MCAGate 模块。
        Args:
            k_size: 卷积核大小
            pool_types: 池化类型。'avg': 平均池化, 'max': 最大池化, 'std': 标准差池化。
        """
        super(MCAGate, self).__init__()

        # 初始化池化层列表
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))  # 添加自适应平均池化层
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))  # 添加自适应最大池化层
            elif pool_type == 'std':
                self.pools.append(StdPool())  # 添加标准差池化层
            else:
                raise NotImplementedError

        # 创建卷积核
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)

        # 实例化 Sigmoid 激活函数的实例
        self.sigmoid = nn.Sigmoid()

        # 创建一个包含两个随机数的张量，并将其封装为一个可学习的参数。 修改一下 创建3个
        self.weight = nn.Parameter(torch.rand(3))


    def forward(self, x):
        # 对每个输入的x进行池化操作，并将处理结果存储到feats中
        feats = [pool(x) for pool in self.pools]

        # 如果只有一个池化结果，则直接使用该结果
        if len(feats) == 1:
            out = feats[0]
        # 如果有两个池化结果，则进行加权相加
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        elif len(feats) == 3:
            weight = torch.sigmoid(self.weight)
            out = 1 / 3 * (feats[0] + feats[1]+feats[2]) + weight[0] * feats[0] + weight[1] * feats[1] + weight[2]*feats[2]
        else:
            assert False, "Feature Extraction Exception!"

        # 对输出进行维度调整
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        # 使用 Sigmoid 激活函数对结果进行处理
        out = self.sigmoid(out)

        # 将处理后的结果扩展为与输入张量x相同的形状
        out = out.expand_as(x)

        # 返回经过门控操作的结果
        return x * out


class MCAlayer_g(nn.Module):
    def __init__(self, inp, no_spatial=True):
        """构造一个 MCA 模块。
        Args:
            inp: 输入特征图的通道数
            no_spatial: 是否构建通道维度之间的交互 True就是不对通道进行自注意力机制
        """
        super(MCAlayer_g, self).__init__()

        # 根据输入通道数计算卷积核大小 这段代码的目的是根据输入张量的通道数，计算出一个合适的卷积核大小 kernel，以适应输入张量的尺寸
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        # 创建高度维度的自注意力门控模块
        # 这里的参数 3 是用于指定自注意力门控模块中的卷积核大小。具体来说，这个卷积核的大小为 3x3。
        # 在自注意力门控模块中，这个卷积核用于处理输入特征图的高度维度，从而实现自注意力操作。
        self.h_cw = MCAGate(3)
        # 创建宽度维度的自注意力门控模块
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        # 如果不仅对通道维度进行自注意力操作，则创建通道维度的自注意力门控模块
        if not no_spatial:
            self.c_hw = MCAGate(kernel)
        # 创建可学习的权重参数
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(1.0))
    def forward(self, x): #x 是输入到 MCALayer 的特征张量，它是一个四维张量，通常表示为 (batch_size, channels, height, width)。
        # 将输入张量x的高度维度与通道维度互换
        x_h = x.permute(0, 2, 1, 3).contiguous()
        # 使用高度维度的自注意力门控模块处理互换后的张量x_h
        x_h = self.h_cw(x_h)
        # 再次将处理后的张量x_h的维度互换回原来的顺序
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        # 将输入张量x的宽度维度与通道维度互换
        x_w = x.permute(0, 3, 2, 1).contiguous()
        # 使用宽度维度的自注意力门控模块处理互换后的张量x_w
        x_w = self.w_hc(x_w)
        # 再次将处理后的张量x_w的维度互换回原来的顺序
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        # 如果不仅对通道维度进行自注意力操作，则使用通道维度的自注意力门控模块处理输入张量x
        if not self.no_spatial:
            x_c = self.c_hw(x)
            # 将三个方向的自注意力结果取平均
            x_out = 1 / 3 * (x_c + x_h + x_w)
            #进行动态调整
            # x_out = self.weight1*x_c + self.weight2*x_h +self.weight3*x_w
        else:
            # 如果仅对高度和宽度维度进行自注意力操作，则将两个方向的自注意力结果取平均
            x_out = 1 / 2 * (x_h + x_w)
            #进行动态调整权重
            # x_out = self.weight1 * x_h + self.weight2 * x_w

        # 返回处理后的张量作为输出
        return x_out










class C2fWithMCA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, threshold1=0.34):
        """初始化CSP块，参数包括输入通道数c1，输出通道数c2，重复次数n，是否使用快捷连接shortcut，分组数g，扩展因子e。"""
        super().__init__()
        # # 初始化阈值 动态调整
        self.thresholds = nn.Parameter(torch.tensor(threshold1))  # 定义动态阈值参数
        # self.thresholds = [threshold1,threshold2]
        self.c = int(c2 * e)  # 隐藏通道数，根据扩展因子和输出通道数计算
        self.mca = MCAlayer_g(3)  # 创建池化通道

        # 创建多个规格的卷积核以适应动态调整的特征图切割 可能切分为1 2 3 4 5，其中1表示不切分
        # 当不切分时
        # self.cv11 = Conv(c1, self.c, 1, 1)
        # self.cv12 = Conv((1 + n) * self.c, c2, 1)
        self.cv11 = Conv(c1, c2, 1, 1)
        # 当切分两份时
        self.cv21 = Conv(c1, 2 * self.c, 1, 1)
        self.cv22 = Conv((2 + n) * self.c, c2, 1)
        # bottleneck模块的卷积次数将不再对其进行改变
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 首先对x进行池化特征提取
        x = self.mca(x)
        # 根据输入动态调整块的数量
        std = x.std().item()
        # 根据标准差和阈值确定分割数量
        if std < self.thresholds:
            splits = 1
        else:
            splits = 2
        if splits == 1:
            y = self.cv11(x)
            # y = self.cv12(y)
        elif splits == 2:
            y = list(self.cv21(x).chunk(splits, 1))
            y.extend(m(y[-1]) for m in self.m)
            y = self.cv22(torch.cat(y, 1))
        return y


    def forward_split(self, x):
        # 首先对x进行池化特征提取
        x = self.mca(x)
        # 根据输入动态调整块的数量
        std = x.std().item()
        # 根据标准差和阈值确定分割数量
        if std < self.thresholds:
            splits = 1
            y = self.cv11(x)
        else:
            splits = 2
            split_sizes = [self.c for _ in range(splits)]  # 计算每个分割块的大小
            y = list(self.cv21(x).split(split_sizes, 1))
            y.extend(m(y[-1]) for m in self.m)
            y = self.cv22(torch.cat(y, 1))
        return y
