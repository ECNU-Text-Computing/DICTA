import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseSA import Mul_HeadSA
from .attention import Attention
from .posEncoding import PositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(self, input_size, num_head, dim_ff, dropout_p, num_layers, norm=None, pad_idx=-1,
                 seq_len=None, use_attention=None, use_sbert=True, use_sbert_seq=True, device='cpu'):
        super(TransformerDecoder, self).__init__()
        # 克隆得到多个encoder layers 论文中默认为6
        self.decoder_layer = DecoderLayer(input_size, num_head, dim_ff, dropout_p)
        self.layers = nn.ModuleList([copy.deepcopy(self.decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.seq_length = seq_len
        self.pad_idx = pad_idx
        self.use_attention = use_attention
        self.use_sbert = use_sbert
        self.use_sbert_seq = use_sbert_seq
        self.fc_input = nn.Linear(1, input_size)
        self.fc_input2 = nn.Linear(input_size, input_size)
        self.fc_sbert = nn.Linear(input_size + 384, input_size)
        self.pos_embedding = PositionalEncoding(d_model=input_size, dropout=dropout_p)
        if use_attention:
            self.attention = Attention(input_size)
        if use_sbert:
            self.cat_size = input_size * 2 + 384 if use_attention else input_size + 384
        else:
            self.cat_size = input_size * 2 if use_attention else input_size
        self.out = nn.Linear(self.cat_size, 1)
        self.device = device
        self.mlp = nn.Linear(384, 1)

    def forward_step(self, input_var, memory, encoder_output, tgt_mask, memory_mask, tgt_key_padding_mask,
                     memory_key_padding_mask, content_embedding):
        # input_var: [src_seq_len, batch_size, 1]
        # content_embedding: [src_seq_len, batch_size, 384]
        # print(input_var.size())
        input_var = self.fc_input(input_var)  # [seq_len, batch_size, input_size]
        input_var = self.pos_embedding(input_var)
        out_len = input_var.size(0)
        if self.use_sbert_seq:
            content_embedding = content_embedding[-out_len:, :, :]  # [trg sent len, batch_size, 384]
            input_var = torch.cat([input_var, content_embedding], dim=2)  # [trg sent len, batch_size, input_size+384]
            input_var = self.fc_sbert(input_var)  # [trg sent len, batch_size, input_size]
        # content_embedding = content_embedding[-out_len:, :, :]
        for mod in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output, decoder_hidden = mod(memory=memory,
                                         tgt=input_var,
                                         tgt_mask=tgt_mask,
                                         memory_mask=memory_mask,
                                         tgt_key_padding_mask=tgt_key_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask)
            # output: [seq_len, batch_size, 1]
            # decoder_hidden: [seq_len, batch_size, input_size]
            input_var = self.fc_input(output)  # [seq_len, batch_size, input_size]

        if self.use_attention:
            if self.use_sbert:
                # output, attn = self.attention(output, encoder_outputs, content_embedding)
                att_output, attn = self.attention(decoder_hidden.permute(1, 0, 2), memory.permute(1, 0, 2))
                # att_output: [batch_size, tgt_seq_len, hidden_size]
                output = torch.cat((att_output, decoder_hidden.permute(1, 0, 2), content_embedding.permute(1, 0, 2)),
                                   dim=2)
            else:
                att_output, attn = self.attention(decoder_hidden.permute(1, 0, 2), memory.permute(1, 0, 2))
                # att_output: [batch_size, tgt_seq_len, hidden_size]
                output = torch.cat((att_output, decoder_hidden.permute(1, 0, 2)), dim=2)
        else:
            if self.use_sbert:
                output = torch.cat([decoder_hidden.permute(1, 0, 2), content_embedding.permute(1, 0, 2)], dim=2)
            else:
                output = decoder_hidden.permute(1, 0, 2)

        output = self.out(output.permute(1, 0, 2))

        return output

    def forward(self, content_embedding, memory, tgt=None, teacher_forcing_ratio=0, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        :param tgt: 解码部分的输入，形状为 [tgt_len,batch_size, input_dim]
        :param memory: 编码部分最后一层的输出 [src_len,batch_size, input_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        encoder_output = memory[-1, :, :]
        tgt, batch_size, tgt_len = self._validate_args(tgt, memory, encoder_output, teacher_forcing_ratio)

        # tgt = torch.stack([content_embedding for i in range(self.seq_length)], dim=0)
        # tgt = self.mlp(tgt)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []

        if use_teacher_forcing:
            decoder_input = tgt
            if self.use_sbert_seq:
                content_embedding = content_embedding.permute(1, 0, 2)
            elif self.use_sbert:
                content_embedding = torch.stack([content_embedding for i in range(tgt_len)], dim=0)
            decoder_output = self.forward_step(decoder_input, memory, encoder_output,
                                               tgt_mask, memory_mask, tgt_key_padding_mask,
                                               memory_key_padding_mask, content_embedding)
            # decoder_output: [tgt_len, batch_size, 1]
            for di in range(decoder_output.size(0)):
                step_output = decoder_output[di, :, :]
                decoder_outputs.append(step_output)
        else:
            decoder_input = tgt[0, :, :].unsqueeze(0)  # [1, batch_size, 1]
            if self.use_sbert:
                content_embedding = content_embedding.unsqueeze(0)
            for di in range(tgt_len):
                if self.use_sbert_seq:
                    content = content_embedding[:, :, di - tgt_len, :]
                else:
                    content = content_embedding
                decoder_output = self.forward_step(decoder_input, memory, encoder_output,
                                                   tgt_mask, memory_mask, tgt_key_padding_mask,
                                                   memory_key_padding_mask, content)
                step_output = decoder_output.squeeze(1)  # [batch_size, 1]
                decoder_outputs.append(step_output)
                decoder_input = decoder_output  # [batch_size, 1, 1]

        return decoder_outputs

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                # inputs = self.norm(inputs)
                batch_size = inputs.size(1)
            else:
                batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.FloatTensor([self.pad_idx] * batch_size).view(1, batch_size, 1).to(self.device)
            if self.seq_length is None:
                seq_length = 5
            else:
                seq_length = self.seq_length
        else:
            seq_length = inputs.size(0)

        return inputs, batch_size, seq_length


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
        tgt1 = self.self_attn(tgt, tgt, tgt,  # [tgt_len, batch_size, input_dim]
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 解码部分输入序列之间'的多头注意力（也就是论文结构图中的Masked Multi-head attention)

        tgt = tgt + self.dropout1(tgt1)  # 接着是残差连接
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
        hidden = self.norm3(tgt)  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len, batch_size, input_dim]
        output = self.fc_output(hidden)
        return output, hidden
        # output: [tgt_len, batch_size, 1]
        # hidden: [tgt_len, batch_size, input_dim]
