import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.atten import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.cnn import enCNN,deCNN


class MSTFormer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, prob_use,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(MSTFormer, self).__init__()
        self.pred_len = out_len
        self.attn = attn  # attention的机制，选择论文提出的prob的机制
        self.output_attention = output_attention  # decode时是否使用attention，不使用就是生成式

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # CNN
        self.encnn = enCNN(seq_len, d_model)
        self.decnn = deCNN(label_len+out_len, d_model)

        #device
        self.device= device

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(l,
                    AttentionLayer(Attn(l, False, factor, attention_dropout=dropout, output_attention=output_attention,device=self.device),
                                   d_model, n_heads, mix=False,device=self.device),  # probsparse attn factor
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,  # 如果distil就把结果卷积小
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(l,
                    AttentionLayer(Attn(l, True, factor, attention_dropout=dropout, output_attention=False,device=self.device),
                                   d_model, n_heads, mix=mix,device=self.device),  # probsparse attn factor
                    AttentionLayer(FullAttention(l, False, factor, attention_dropout=dropout, output_attention=False,device=self.device),
                                   d_model, n_heads, mix=False,device=self.device),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_map, y_map, x_dec, x_mark_dec, # mark 是时间
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        x_map = self.encnn(x_map)
        enc_out, attns, x_map = self.encoder(enc_out, x_map,x_enc, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        y_map = self.decnn(y_map)
        dec_out = self.decoder(dec_out, enc_out, y_map,x_dec, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
