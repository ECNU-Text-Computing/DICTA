import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseSA import Mul_HeadSA
from .posEncoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_head, dim_ff, dropout_p, num_layers,
                 use_sbert=True, use_sbert_seq=True, norm=None):
        super(TransformerEncoder, self).__init__()
        # 克隆得到多个encoder layers 论文中默认为6
        self.encoder_layer = EncoderLayer(input_size, num_head, dim_ff, dropout_p)
        self.layers = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.fc_input = nn.Linear(1, input_size)
        self.fc_input2 = nn.Linear(input_size, input_size)
        self.fc_sbert = nn.Linear(input_size + 384, input_size)
        self.pos_embedding = PositionalEncoding(d_model=input_size, dropout=dropout_p)
        self.use_sbert = use_sbert
        self.use_sbert_seq = use_sbert_seq
        self.mlp = nn.Linear(384, input_size)

    def forward(self, src, content, mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:# [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        # src: [batch_size, seq_len, 1]
        output = self.fc_input(src)
        # output = torch.stack([content for i in range(5)], dim=0)
        # output = self.mlp(output)
        output = self.pos_embedding(output)
        in_len = output.size(0)
        if self.use_sbert_seq:
            input_var = torch.cat([output, content[:, :in_len, :].permute(1, 0, 2)], dim=2)
            # [seq_len, batch_size, hidden_size+384]
            output = self.fc_sbert(input_var)  # [seq_len, batch_size, hidden_size]
        # input = self.fc_input2(input)
        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)  # 多个encoder layers层堆叠后的前向传播过程
        if self.norm is not None:
            output = self.norm(output)
        return output, content
        # output: [src_len, batch_size, num_heads * kdim] <==> [src_len, batch_size, embed_dim]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(EncoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        """
        self.self_attn = Mul_HeadSA(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len, batch_size, embed_dim]
        :param src_mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  # 计算多头注意力
        # src2: [src_len, batch_size, num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # [src_len,batch_size,num_heads*kdim]

        src2 = self.activation(self.linear1(src))  # [src_len,batch_size,dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len,batch_size,num_heads*kdim]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
