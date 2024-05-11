import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import Attention


class DecoderCNN(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device, seq_len=None,
                 use_attention=False, use_sbert=True, use_sbert_seq=True):
        super(DecoderCNN, self).__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.device = device
        self.seq_length = seq_len
        self.use_attention = use_attention
        self.use_sbert = use_sbert
        self.use_sbert_seq = use_sbert_seq

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.fc_input = nn.Linear(1, hid_dim)
        # self.pos = nn.Linear(100, output_dim)
        self.fc_sbert = nn.Linear(hid_dim + 384, hid_dim)

        self.fc_en = nn.Linear(hid_dim, output_dim)

        self.hid = nn.Linear(output_dim, hid_dim)

        self.attn_hid2 = nn.Linear(hid_dim, output_dim)
        self.attn_2hid = nn.Linear(hid_dim, hid_dim)
        self.mlp = nn.Linear(384, 1)

        if use_sbert:
            self.cat_size = hid_dim * 2 + 384 if use_attention else hid_dim + 384
        else:
            self.cat_size = hid_dim * 2 if use_attention else hid_dim
        self.out = nn.Linear(self.cat_size, 1)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        if use_attention:
            self.attention = Attention(self.hid_dim)

    def forward_step(self, input_var, encoder_conved, encoder_combined, content_embedding):
        # input_var: [batch_size, trg sent len, 1]
        # encoder_conved: [batch size, src sent len, hid dim]
        # content_embedding: [batch size, src sent len, 384]
        input_var = self.fc_input(input_var)  # [batch_size, trg sent len, hid dim]
        input_var = self.dropout(input_var)
        out_len = input_var.size(1)
        if self.use_sbert_seq:
            content_embedding = content_embedding[:, -out_len:, :]  # [batch_size, trg sent len, 384]
            input_var = torch.cat([input_var, content_embedding], dim=2)  # [batch_size, trg sent len, hid dim+384]
            input_var = self.fc_sbert(input_var)  # [batch_size, trg sent len, hid_dim]
        # content_embedding = content_embedding[:, -out_len:, :]
        conv_input = input_var  # [batch_size, tgt sent len, hid_dim]
        conv_input = conv_input.permute(0, 2, 1)  # [batch_size, hidden_dim, tgt sent len]

        for i, conv in enumerate(self.convs):
            padding = torch.zeros(conv_input.shape[0], conv_input.shape[1], self.kernel_size - 1).fill_(
                self.pad_idx).to(self.device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            conved = conv(padded_conv_input)
            # conved = [batch size, 2 * hid dim, tgt sent len]

            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, trg sent len]
            # 引进辅助任务：分类，加一个cross entro，（交叉熵和NLLLoss）

            att = None
            # if self.use_attention:
            #     attention, conved = self.calculate_attention(input_var, conved, encoder_conved, encoder_combined)
            #     # attention = [batch size, trg sent len, src sent len]
            #     # conved = [batch size, hid dim, trg sent len]
            #     # print('conved 1',conved.shape)
            # else:
            #     conved = conved.permute(0, 2, 1)  # [batch size, hid dim, trg sent len]

            conved = (conved + conv_input) * self.scale  # [batch size, hid dim, trg sent len]
            conv_input = conved
            # hidden = conved.permute(0, 2, 1)  # [batch size, trg sent len, hid dim]

        hidden = self.dropout(conved.permute(0, 2, 1))  # [batch size, trg sent len, hid dim]

        attn = None
        if self.use_attention:
            if self.use_sbert:
                att_output, attn = self.attention(hidden, encoder_conved)
                # att_output: [batch_size, tgt_seq_len, hidden_size]
                output = torch.cat((att_output, hidden, content_embedding), dim=2)
                # [batch_size, tgt_seq_len, cat_size]
            else:
                att_output, attn = self.attention(hidden, encoder_conved)
                # att_output: [batch_size, tgt_seq_len, hidden_size]
                output = torch.cat((att_output, hidden), dim=2)  # [batch_size, tgt_seq_len, cat_size]
        else:
            if self.use_sbert:
                output = torch.cat([hidden, content_embedding], dim=2)  # [batch_size, tgt_seq_len, cat_size]
            else:
                output = hidden

        predicted = self.out(output)  # [batch_size, trg sent len, 1]

        return predicted, hidden, attn

    def forward(self, content_embedding, encoder_conved, encoder_combined, tgt=None, teacher_forcing_ratio=0):
        encoder_output = encoder_conved[:, -1, :]
        tgt, batch_size, tgt_len = self._validate_args(tgt, encoder_conved, encoder_output, teacher_forcing_ratio)

        # tgt = torch.stack([content_embedding for i in range(self.seq_length)], dim=1)
        # tgt = self.mlp(tgt)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        attn_outputs = []

        if use_teacher_forcing:
            decoder_input = tgt
            if self.use_sbert_seq:
                content_embedding = content_embedding
            elif self.use_sbert:
                content_embedding = torch.stack([content_embedding for i in range(tgt_len)], dim=1)
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, encoder_conved,
                                                                     encoder_combined, content_embedding)
            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decoder_outputs.append(step_output)
                attn_outputs.append(step_attn)
        else:
            decoder_input = tgt[:, 0, :].unsqueeze(1)  # [batch_size, 1, 1]
            if self.use_sbert:
                content_embedding = content_embedding.unsqueeze(1)
            for di in range(tgt_len):
                if self.use_sbert_seq:
                    content = content_embedding[:, :, di - tgt_len, :]
                else:
                    content = content_embedding
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, encoder_conved,
                                                                              encoder_combined, content)
                step_output = decoder_output.squeeze(1)  # [batch_size, 1]
                decoder_outputs.append(step_output)
                attn_outputs.append(step_attn)
                decoder_input = decoder_output  # [batch_size, 1, 1]

        return decoder_outputs, attn_outputs

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
                batch_size = inputs.size(0)
            else:
                batch_size = encoder_hidden.size(0)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.FloatTensor([self.pad_idx] * batch_size).view(batch_size, 1, 1).to(self.device)
            # if torch.cuda.is_available():
            #     inputs = inputs.cuda()
            # inputs = inputs.to(device)
            if self.seq_length is None:
                seq_length = 5
            else:
                seq_length = self.seq_length
        else:
            seq_length = inputs.size(1)

        return inputs, batch_size, seq_length
