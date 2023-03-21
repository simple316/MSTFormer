import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, layer_num, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 device='cuda:0'):
        super(FullAttention, self).__init__()
        self.layer_num = layer_num  # 用来判断是不是第一层，如果是第一层attention进行选择计算
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.divice = device

    def forward(self, queries, keys, values, atten_data, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, layer_num, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 device='cuda:0'):
        super(ProbAttention, self).__init__()
        self.layer_num = layer_num  # 用来判断是不是第一层，如果是第一层attention进行选择计算
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.device = device

    def data_normal_2d(self, orign_data, dim="col"):
        """
        针对于2维tensor归一化
        可指定维度进行归一化，默认为行归一化
        参数1为原始tensor，参数2为默认指定行，输入其他任意则为列
        """
        if dim == "col":
            dim = 1
            d_min = torch.min(orign_data, dim=dim)[0]
            for idx, j in enumerate(d_min):
                if j < 0:
                    orign_data[idx, :] += torch.abs(d_min[idx])
                    d_min = torch.min(orign_data, dim=dim)[0]
        else:
            dim = 0
            d_min = torch.min(orign_data, dim=dim)[0]
            for idx, j in enumerate(d_min):
                if j < 0:
                    orign_data[idx, :] += torch.abs(d_min[idx])
                    d_min = torch.min(orign_data, dim=dim)[0]
        d_max = torch.max(orign_data, dim=dim)[0]
        dst = d_max - d_min
        if d_min.shape[0] == orign_data.shape[0]:
            d_min = d_min.unsqueeze(1)
            dst = dst.unsqueeze(1)
        else:
            d_min = d_min.unsqueeze(0)
            dst = dst.unsqueeze(0)
        norm_data = torch.sub(orign_data, d_min).true_divide(dst)
        return norm_data

    def _sort_importance(self, x, u, H):
        b, l, _ = x.shape
        x_abs = torch.abs(x)
        index = 1 / torch.log(torch.flip(torch.arange(3, l + 3), dims=[0])).expand(b, l).to(
            self.device)  # 按照位置计算权重，预测点权重最高，越往前越低

        # 沿着指定的维度, 重复张量的元素
        mean_COG = torch.repeat_interleave(torch.mean(x_abs[:, :, 1:], 1).unsqueeze(1), l, dim=1)
        mean_SOG = torch.repeat_interleave(torch.mean(x_abs[:, :, :1], 1).unsqueeze(1), l, dim=1)
        x_abs[:, :, 1:] = torch.clamp((x_abs[:, :, 1:] - mean_COG), max=1, min=0.0)
        x_abs[:, :, :1] = torch.clamp((x_abs[:, :, :1] - mean_SOG), max=1, min=0.0)
        x_abs = x_abs + 1
        index_ = self.data_normal_2d(torch.log(x_abs[:, :, 1:]).squeeze(2))  # 按照方向差计算权重，转向越多，权重越高
        index = torch.stack((index, index_), 1)  # index[:, 1:2, :] 储存方向差的权重

        index_ = self.data_normal_2d(torch.log(x_abs[:, :, :1]).squeeze(2))
        index = torch.cat((index, index_.unsqueeze(1)), 1)  # index[:, 2:3, :] 储存速度差的权重
        index = torch.cat((index, (index[:, :1, :] + index[:, 1:2, :] + index[:, 2:3, :])), 1)
        # index = torch.cat((index, 0.5*index[:,:1,:]+0.5*index[:,1:2,:]+index[:,2:3,:]), 1)
        # index = torch.cat((index, 0.5*index[:,:1,:]+index[:,1:2,:]+0.5*index[:,2:3,:]), 1)
        # index = torch.cat((index, index[:,:1,:]+0.5*index[:,1:2,:]+0.5*index[:,2:3,:]), 1)
        weight = nn.Parameter(torch.rand(H - index.shape[1], 2).to(self.device))
        for i in range(len(weight)):
            index = torch.cat(
                (index, index[:, :1, :] + weight[i][0] * index[:, 1:2, :] + weight[i][1] * index[:, 2:3, :]), 1)
        # while index.shape[1]<H:
        #     index = torch.cat((index, index[:,:1,:] + torch.rand(1).to(self.device) * index[:, 1:2, :] + torch.rand(1).to(self.device) * index[:, 2:3, :]), 1)
        # weight = nn.Parameter(torch.rand(H, 2).to(self.device))
        # for i in range(H):
        #     index = torch.cat(
        #         (index, index[:, :1, :] + weight[i][0] * index[:, 1:2, :] + weight[i][1] * index[:, 2:3, :]), 1)
        values, index_ = torch.sort(index, dim=2, descending=True)
        return index_[:, -H:, :u]

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # expand()将数据扩为更大的维度,unsqueeze(-3)倒数第三位插入维度
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        # randint(low=0, high, size, out=None, dtype=None)随机生成【0,72】内数字，大小为（72,15）
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        # max(-1)中的-1表示按照最后一个维度（行）求最大值,方括号[1]则表示返回最大值的索引
        # torch.div张量和标量做逐元素除法或者两个可广播的张量之间做逐元素除法
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # [30, 8, 72]
        M_top = M.topk(n_top, sorted=False)[1]  # [30, 8, 15]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)  #[30, 8, 15, 64]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k  #K.transpose[30, 8, 64, 72]
        # Q_K [30, 8, 15, 64]
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # 先把96个v都用均值代替
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) [30, 8, 15, 72]

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 对25个有Q的更新V，其余的还是没变 [30,8,72,64]
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, atten_data, attn_mask):
        B, L_Q, H, D = queries.shape  # [30, 72, 8, 64]
        _, L_K, _, _ = keys.shape

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k) np.ceil计算大于等于该值的最小整数
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        queries = queries.transpose(2, 1)  # [30, 8, 72, 64]
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        if self.layer_num == 0:  # 第一层根据规则计算，挑选计算attention的Q
            index = self._sort_importance(atten_data, u, H)
            Q_reduce = queries[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index,
                       :]  # [30, 8, 15, 64]
            scores_top = torch.matmul(Q_reduce, keys.transpose(-2, -1))  # keys.transpose(-2, -1) [30, 8, 64, 72]
            # scores_top [30, 8, 15, 72]
        else:
            scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False, device='cuda:0'):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.device = device

    def forward(self, queries, keys, values, atten_data, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            atten_data,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
