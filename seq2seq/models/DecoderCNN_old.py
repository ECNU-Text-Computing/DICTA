import random
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


class DecoderCNN(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device):
        super(DecoderCNN, self).__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok = nn.Linear(1, output_dim)
        self.fc_input = nn.Linear(output_dim, output_dim)
        # self.pos = nn.Linear(100, output_dim)

        self.fc_en = nn.Linear(hid_dim, output_dim)

        self.hid = nn.Linear(output_dim, hid_dim)

        self.attn_hid2 = nn.Linear(hid_dim, output_dim)
        self.attn_2hid = nn.Linear(hid_dim, hid_dim)

        self.out = nn.Linear(hid_dim, 1)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, att_input, conved, encoder_conved, encoder_combined):
        conved_input = self.attn_hid2(conved.permute(0, 2, 1))
        # conved_input = [batch size, trg sent len, output dim]

        combined = self.hid(att_input +conved_input)
        combined = combined * self.scale
        # combined = [batch size, trg sent len, hidden dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        # energy = [batch size, trg sent len, src sent len]

        attention = F.softmax(energy, dim=2)
        # attention = [batch size, trg sent len, src sent len]

        attended_encoding = torch.matmul(attention, (encoder_conved + encoder_combined))
        # attended_encoding = [batch size, trg sent len, hid dim]
        attended_encoding = self.attn_2hid(attended_encoding)

        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        # attended_combined = [batch size, hid dim, trg sent len]

        return attention, attended_combined

    def forward(self, encoder_conved, encoder_combined, tgt=None, teacher_forcing_ratio=0):
        # tgt = [batch size, trg sent len, 1]
        # pos = [batch size, trg sent len, 1]
        # encoder_conved = encoder_combined = [batch size, src sent len, hidden dim]
        # pos = torch.arange(0, tgt.shape[1]).unsqueeze(0).repeat(tgt.shape[0], 1).to(device)
        if tgt is None:
            teacher_forcing_ratio = 0
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            tok = self.tok(tgt)
            # tok = self.fc_input(tok)
            # tok = [batch size, trg sent len, output dim]


        else:
            tok = self.fc_en(encoder_conved)
            # tok = self.fc_input(tok)
            # tok = [batch size, src sent len, output dim]

        input = self.dropout(tok)
        # input = [batch size, trg sent len, output dim]
        conv_input = self.hid(input)
        # conv_input = [batch size, trg sent len, hid dim]

        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, trg sent len]



        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(conv_input.shape[0], conv_input.shape[1], self.kernel_size - 1).fill_(
                self.pad_idx).to(device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            conved = conv(padded_conv_input)
            # conved = [batch size, 2*hid dim, trg sent len]

            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, trg sent len]

            attention, conved = self.calculate_attention(input, conved, encoder_conved, encoder_combined)
            # attention = [batch size, trg sent len, src sent len]
            # conved = [batch size, hid dim, trg sent len]
            # print('conved 1',conved.shape)

            conved = (conved + conv_input) * self.scale
            conv_input = conved
            # print('conved 2', conved.shape)

        output = self.out(conved.permute(0, 2, 1))  # [batch size, trg sent len, 1]
        output = output.permute(1, 0, 2)  # [trg sent len, batch size, 1]
        return output, attention