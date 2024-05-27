import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#动态调整卷积核位置的class
# class DynamicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding=0):
#         super(DynamicConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.padding = padding
#
#         self.conv_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
#         self.conv_bias = nn.Parameter(torch.Tensor(out_channels))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
#         if self.conv_bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.conv_bias, -bound, bound)
#
#     def forward(self, x):
#         variance = torch.var(x, dim=(2, 3), keepdim=True)  # Compute variance along H and W dimensions
#         mask = variance > 0.5  # Define a threshold for variance
#         x = F.conv2d(x, self.conv_weight * mask, bias=self.conv_bias, padding=self.padding)
#         return x
#动态根据方差调整权重的
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv_bias, -bound, bound)

    def forward(self, x):
        # 计算每个通道的方差，并扩展到与权重相同的维度
        variance = torch.var(x, dim=(2, 3), keepdim=True)  # 计算方差
        variance = variance.expand_as(self.conv_weight)  # 扩展方差的尺寸以匹配权重

        # 根据方差调整权重
        adjusted_weight = self.conv_weight * variance

        # 应用调整后的权重进行卷积
        x = F.conv2d(x, adjusted_weight, self.conv_bias, self.stride, self.padding, self.dilation)
        return x

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = self.fc2(out)
        alpha = torch.softmax(out, dim=1)
        return alpha

class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=True):
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = DynamicConv2d(1, 1, kernel_size=3)  # DynamicConv2d for height dimension
        self.w_hc = DynamicConv2d(1, 1, kernel_size=3)  # DynamicConv2d for width dimension
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = DynamicConv2d(kernel, 1, kernel_size=3)  # DynamicConv2d for channel dimension
            self.attention = Attention(3, 1)  # Attention mechanism for dynamic weight adjustment

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            # Apply attention mechanism
            alpha_h = self.attention(x_h)
            alpha_w = self.attention(x_w)
            alpha_c = self.attention(x_c)
            # Weight each branch output with attention weights
            x_h = x_h * alpha_h
            x_w = x_w * alpha_w
            x_c = x_c * alpha_c
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            # Apply attention mechanism
            alpha_h = self.attention(x_h)
            alpha_w = self.attention(x_w)
            # Weight each branch output with attention weights
            x_h = x_h * alpha_h
            x_w = x_w * alpha_w
            x_out = 1 / 2 * (x_h + x_w)

        return x_out