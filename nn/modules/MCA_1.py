import torch
from torch import nn
import math

__all__ = ['MCALayer', 'MCAGate']


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1)) #添加自适应平均池化层
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))#添加自适应最大池化层
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
        #创建卷积核
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        #实例化Sigmoid 激活函数的实例
        self.sigmoid = nn.Sigmoid()
        #创建了一个包含两个随机数的张量，并将其封装为一个可学习的参数。
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools] #对每个输入的x进行池化池中的池化操作 并将处理结果存储到feats中

        if len(feats) == 1:
            out = feats[0]#如果只有一个 就是正常输出
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1] #如果有两个池化操作的话 就加权相加
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        # 使用permute方法对张量out进行维度重新排列，将原来的第0维调整到新的第0维，第3维调整到新的第1维，第2维调整到新的第2维，第1维调整到新的第3维。
        # 使用contiguous方法使得重新排列后的张量在内存中是连续存储的，确保后续操作的正确性
        out = self.conv(out) #对池化完的数据进行卷积
        out = out.permute(0, 3, 2, 1).contiguous() #再次更换位置

        out = self.sigmoid(out) #使用sigmoid激活函数对变量out进行处理
        out = out.expand_as(x)#将处理后的out扩展为与输入变量x相同的形状。

        # 在深度学习中，卷积神经网络（CNN）通常处理的是四维张量，表示为(batch_size, channels, height,
        #                                                           width)。在处理图像数据时，通常会按照(batch_size, height,
        #                                                                                              width, channels)
        # 的顺序进行处理，但在PyTorch中，卷积操作默认的输入格式是(batch_size, channels, height, width)。
        #
        # 因此，在进行卷积操作之前，需要将张量的维度重新排列，以符合PyTorch卷积操作的输入要求。所以在代码中，先使用permute方法将维度重新排列为(
        #     batch_size, width, height, channels)，然后进行卷积操作，最后再次使用permute方法将维度重新排列为(batch_size,
        #                                                                                                   channels,
        #                                                                                                   height,
        #                                                                                                   width)。这样可以确保卷积操作能够正确处理输入数据，并保持后续操作的正确性。
        #


        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=True):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3) #对高度维度进行池化 并且加以权重
        self.w_hc = MCAGate(3) #对宽度维度进行池化 并且加以权重
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h) #将新的张量x_h传入self.h_cw函数中进行处理
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w) #将新的张量x_w传入self.w_hc函数中进行处理
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out