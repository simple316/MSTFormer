import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, layer_num, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.layer_num = layer_num
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # 一个方向卷积，卷完为N*1，再进行卷积核大小为n的maxpolling，变成一个数
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_map, atten_x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x, atten_x,
            attn_mask=attn_mask
        )

        x = x + self.dropout(new_x)

        if self.layer_num == 0:
            z = self.dropout(x_map)
            x = self.norm1(x)
            y = x + z
            y = self.norm2(y)
        else:
            y = x = self.norm3(x)
        # FNN
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # transpose矩阵转制
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm4(x + y), attn, x_map


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        # nn.ModuleList专门储存model的list
        # 继承了nn.Module的网络模型class可以使用nn.ModuleList并识别其中的parameters；
        # ModuleList中的子module能够被主module所识别并调整参数
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, x_map, atten_x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn, x_map = attn_layer(x, x_map, atten_x, attn_mask=attn_mask)
                x = conv_layer(x)
                x_map = conv_layer(x_map)
                attns.append(attn)
            x, attn, x_map = self.attn_layers[-1](x, x_map, atten_x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, x_map, atten_x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, x_map

