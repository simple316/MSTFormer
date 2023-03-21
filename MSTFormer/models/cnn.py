import ipdb
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

class enCNN(nn.Module):
    def __init__(self,seq_len,d_model):
        super(enCNN,self).__init__()
        self.channel=seq_len
        self.d_model=d_model
        # nn.Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数
        # nn.Conv2d 二维卷积 先实例化再使用 在Pytorch的nn模块中，它是不需要你手动定义网络层的权重和偏置的
        self.conv1 = nn.Sequential(  # input shape (1,28,28)
            nn.Conv3d(in_channels=1,  # input height 必须手动提供 输入张量的channels数 [N, C, H, W]
                      out_channels=1,  # n_filter 必须手动提供 输出张量的channels数
                      kernel_size=(1, 3, 3),  # filter size 必须手动提供 卷积核的大小
                      # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
                      stride=1,  # filter step 卷积核在图像窗口上每次平移的间隔，即所谓的步长
                      padding=(0,1,1)  # con2d出来的图片大小不变 Pytorch与Tensorflow在卷积层实现上最大的差别就在于padding上
                      ),  # output shape (16,28,28) 输出图像尺寸计算公式是唯一的 # O = （I - K + 2P）/ S +1
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,2,2))
            # 2x2采样，28/2=14，output shape (16,14,14) maxpooling有局部不变性而且可以提取显著特征的同时降低模型的参数，从而降低模型的过拟合
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=16,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),  #depth，h，w
                      padding="same"),
            #三维卷积层, 输入数据的尺度是(N, C_in,D,H,W)
            #(1，3，100，1000，1000）表示输入1个视频，每个视频中图像的通道数是3，每个视频中包含的图像数是100，图像的大小是1000 x 1000。
            #输出尺度（N,C_out,D_out,H_out,W_out）
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool3d(kernel_size=(3, 2, 2), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16,
                      out_channels=self.channel,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),  # depth，h，w
                      padding="same"),
            # 三维卷积层, 输入数据的尺度是(N, C_in,D,H,W)
            # (1，3，100，1000，1000）表示输入1个视频，每个视频中图像的通道数是3，每个视频中包含的图像数是100，图像的大小是1000 x 1000。
            # 输出尺度（N,C_out,D_out,H_out,W_out）
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)
        )
        self.out = nn.Linear(252, 512)  #72,24:252;48,12:189;    3:840 新map252 大map 378
        #改变map大小：[all:1134,1/2:567,1/3:378,1/4:252,1/5:189]
    def forward(self,x):
        x=x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x =  x.view(x.size(0),x.size(1), -1)  # flat (batch_size, 32*7*7)
        # 将前面多维度的tensor展平成一维 x.size(0)指batchsize的值
        # view()函数的功能根reshape类似，用来转换size大小
        x = self.out(x)  # fc out全连接层 分类器
        return x


class deCNN(nn.Module):
    def __init__(self,seq_len,d_model):
        super(deCNN,self).__init__()
        self.channel=seq_len
        self.d_model=d_model
        # nn.Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数
        # nn.Conv2d 二维卷积 先实例化再使用 在Pytorch的nn模块中，它是不需要你手动定义网络层的权重和偏置的
        self.conv1 = nn.Sequential(  # input shape (1,28,28)
            nn.Conv3d(in_channels=1,  # input height 必须手动提供 输入张量的channels数 [N, C, H, W]
                      out_channels=1,  # n_filter 必须手动提供 输出张量的channels数
                      kernel_size=(1, 3, 3),  # filter size 必须手动提供 卷积核的大小
                      # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
                      stride=1,  # filter step 卷积核在图像窗口上每次平移的间隔，即所谓的步长
                      padding=(0,1,1)  # con2d出来的图片大小不变 Pytorch与Tensorflow在卷积层实现上最大的差别就在于padding上
                      ),  # output shape (16,28,28) 输出图像尺寸计算公式是唯一的 # O = （I - K + 2P）/ S +1
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,2,2))
            # 2x2采样，28/2=14，output shape (16,14,14) maxpooling有局部不变性而且可以提取显著特征的同时降低模型的参数，从而降低模型的过拟合
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=16,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),  #depth，h，w
                      padding="same"),
            #三维卷积层, 输入数据的尺度是(N, C_in,D,H,W)
            #(1，3，100，1000，1000）表示输入1个视频，每个视频中图像的通道数是3，每个视频中包含的图像数是100，图像的大小是1000 x 1000。
            #输出尺度（N,C_out,D_out,H_out,W_out）
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool3d(kernel_size=(3, 2, 2), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16,
                      out_channels=self.channel,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),  # depth，h，w
                      padding="same"),
            # 三维卷积层, 输入数据的尺度是(N, C_in,D,H,W)
            # (1，3，100，1000，1000）表示输入1个视频，每个视频中图像的通道数是3，每个视频中包含的图像数是100，图像的大小是1000 x 1000。
            # 输出尺度（N,C_out,D_out,H_out,W_out）
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)
        )
        self.out = nn.Linear(63, 512)  #72,24:63;48,12:189;  3:280 新map：63 大map 126
        # 改变map大小：[all:378,1/2:189,1/3:126,1/4:63,1/5:63]
    def forward(self,x):
        x=x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x =  x.view(x.size(0),x.size(1), -1)  # flat (batch_size, 32*7*7)
        # 将前面多维度的tensor展平成一维 x.size(0)指batchsize的值
        # view()函数的功能根reshape类似，用来转换size大小
        x = self.out(x)  # fc out全连接层 分类器
        return x