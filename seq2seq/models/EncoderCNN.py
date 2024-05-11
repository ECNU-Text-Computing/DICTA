import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):

    def __init__(self, input_dim, hid_dim, n_layers, kernel_size, dropout,
                 use_sbert=True, use_sbert_seq=True, device='cpu'):
        super(EncoderCNN, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.use_sbert = use_sbert
        self.use_sbert_seq = use_sbert_seq
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok = nn.Linear(1, input_dim)
        self.fc_input = nn.Linear(input_dim, input_dim)
        self.fc_sbert = nn.Linear(input_dim + 384, hid_dim)
        self.hid = nn.Linear(input_dim, hid_dim)
        self.hid2 = nn.Linear(hid_dim, input_dim)

        # 位置信息
        # self.pos = nn.Linear(100, input_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim, out_channels=2 * hid_dim, kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(384, input_dim)

    def forward(self, src, content):
        # src=[batch_size, src_sent_len, 1]
        # 构造pos张量，就是使用src的格式构建一个相同的batch_size的张量
        # pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)
        # pos=[batch_size, src_sent_len]

        tok = self.tok(src)
        # src = torch.stack([content for i in range(5)], dim=1)
        # tok = self.mlp(src)

        # tok = self.fc_input(tok)
        # pos = self.pos(pos)
        # tok = pos = [batch size, src sent len, input dim]
        in_len = tok.size(1)
        if self.use_sbert_seq:
            input_var = torch.cat([tok, content[:, :in_len, :]], dim=2)  # [batch_size, seq_len, hidden_size+384]
            input = self.fc_sbert(input_var)  # [batch_size, seq_len, hidden_size]
        else:
            input = self.dropout(self.hid(tok))
        # input = self.dropout(self.hid(tok))
        # 通过linear层将嵌入好的数据传入转为hid_dim
        conv_input = input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, src sent len]

        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2*hid dim, src sent len]

            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, src sent len]

            # 传入残差连接
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, src sent len]

            conv_input = conved

        # 使用permute进行转置，将最后一个元素的转为input_dim
        conved = conved.permute(0, 2, 1)
        # conved = [batch size, src sent len, hid dim]

        combined = (conved + input) * self.scale

        return conved, combined, content
