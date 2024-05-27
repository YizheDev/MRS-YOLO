import torch
import torch.nn as nn
import torch.nn.functional as F


# 假设StdPool和MCAGate已经按前面的描述实现
# 这里直接使用这两个类

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

class CSPBottleneckWithMCA(nn.Module):
    def __init__(self, c1, c2, n=1, use_mca=True):
        super(CSPBottleneckWithMCA, self).__init__()
        self.use_mca = use_mca
        c_ = int(c2 // 2)
        # 分割特征图，一半通过Bottleneck，另一半直接通过
        self.conv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.conv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        self.mca = MCAGate(3) if use_mca else nn.Identity()

        self.bottlenecks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_),
                nn.ReLU(inplace=True)
            ) for _ in range(n)]
        )

        self.final_conv = nn.Conv2d(2 * c_, c2, 1, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        # 将输入分割为两部分
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))

        # 对第一部分应用Bottleneck序列
        x1 = self.bottlenecks(x1)

        # 可选：在两部分上应用MCA门控
        if self.use_mca:
            x1 = self.mca(x1)
            x2 = self.mca(x2)

        # 合并两部分特征
        x = torch.cat([x1, x2], dim=1)
        x = self.final_bn(self.final_conv(x))
        return F.relu(x)


# 示例：创建一个具有特定输入输出通道的模块，并启用MCA
model = CSPBottleneckWithMCA(c1=512, c2=1024, n=2, use_mca=True)
print(model)

# 假设有一个输入
x = torch.rand(1, 512, 32, 32)  # 示例输入：batch_size=1, channels=512, height=32, width=32
output = model(x)
print(output.shape)  # 查看输出形状，期望输出通道为1024
