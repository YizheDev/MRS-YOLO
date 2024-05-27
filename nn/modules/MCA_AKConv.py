import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange


class StdPool(nn.Module):
    def forward(self, x):
        # 标准差池化实现，对每个通道计算标准差
        return torch.std(x, dim=-1, keepdim=True)


class AKConvMCA(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1):
        super().__init__()
        self.num_param = num_param
        self.stride = stride

        # AKConv 参数生成网络
        self.offset_gen = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)

        # MCA 注意力生成网络
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.std_pool = StdPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # AKConv的动态调整卷积核
        self.dynamic_conv = nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), groups=inc,
                                      bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.silu = nn.SiLU()

    def forward(self, x):
        offset = self.offset_gen(x)

        # AKConv的动态调整卷积核采样操作（简化示意，具体根据需要调整）
        # 省略动态调整卷积核的具体实现细节，这里只是一个简化的占位符
        dynamic_feature = self.dynamic_conv(x)
        dynamic_feature = self.bn(dynamic_feature)
        dynamic_feature = self.silu(dynamic_feature)

        # MCA 注意力机制
        avg_pooled = self.avg_pool(dynamic_feature)
        std_pooled = self.std_pool(dynamic_feature)
        combined = torch.cat([avg_pooled, std_pooled], dim=1)
        attn = self.conv(combined)
        attn = self.sigmoid(attn)

        # 应用注意力
        attn = attn.expand_as(dynamic_feature)
        out = dynamic_feature * attn

        return out

# 注意：这段代码简化了AKConv的动态卷积核采样步骤，实际应用中可能需要根据具体的动态采样算法来实现。