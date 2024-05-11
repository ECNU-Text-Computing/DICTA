import copy
import random
import torch.nn as nn
import torch.nn.functional as F

from .baseSA import Mul_HeadSA

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, num_head, dim_ff, dropout_p, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        # 克隆得到多个encoder layers 论文中默认为6
        self.decoder_layer = DecoderLayer(input_size, num_head, dim_ff, dropout_p)
        self.layers = nn.ModuleList([copy.deepcopy(self.decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.fc_input = nn.Linear(1, input_size)
        self.fc_input2 = nn.Linear(input_size, input_size)

    def forward(self, memory, tgt=None, teacher_forcing_ratio=0, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt: 解码部分的输入，形状为 [tgt_len,batch_size, input_dim]
        :param memory: 编码部分最后一层的输出 [src_len,batch_size, input_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            input = self.fc_input(tgt)
            # input = self.fc_input2(input)
        else:
            input = memory  # [tgt_len, batch_size, input_dim]
            # input = self.fc_input2(input)

        for mod in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output = mod(memory=memory,
                         tgt=input,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            input = self.fc_input(output)
        # if self.norm is not None:
        #     output = self.norm(output)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(DecoderLayer, self).__init__()
        """
               :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
               :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
               :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
               :param dropout:         丢弃率，论文中的默认值为 0.1
               """
        self.self_attn = Mul_HeadSA(input_dim=d_model, num_heads=nhead, dropout=dropout)
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        self.multihead_attn = Mul_HeadSA(input_dim=d_model, num_heads=nhead, dropout=dropout)
        # 编码部分输出（memory）和解码部分之间的多头注意力机制。
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.fc_output = nn.Linear(d_model, 1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, memory, tgt, teacher_forcing_ratio=0, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt:  解码部分的输入，形状为 [tgt_len, batch_size, input_dim]
        :param memory: 编码部分的输出（memory）, [src_len, batch_size,input_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        tgt2 = self.self_attn(tgt, tgt, tgt,  # [tgt_len, batch_size, input_dim]
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 解码部分输入序列之间'的多头注意力（也就是论文结构图中的Masked Multi-head attention)


        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)  # [tgt_len, batch_size, input_dim]

        tgt2 = self.multihead_attn(tgt, memory, memory,  # [tgt_len, batch_size, input_dim]
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # 解码部分的输入经过多头注意力后同编码部分的输出（memory）通过多头注意力机制进行交互
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)  # [tgt_len, batch_size, input_dim]

        tgt2 = self.activation(self.linear1(tgt))  # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout(tgt2))  # [tgt_len, batch_size, input_dim]
        # 最后的两层全连接
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len, batch_size, input_dim]
        tgt = self.fc_output(tgt)
        return tgt  # [tgt_len, batch_size, 1]