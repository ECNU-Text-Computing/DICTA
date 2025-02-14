import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2 + 384, dim)
        self.linear_out2 = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, encoder_hidden):
        # content_embedding: [batch_size, out_len, sbert_dim]
        # output: [batch_size, out_len, hidden_size]
        # encoder_hidden: [batch_size, in_len, hidden_size]
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = encoder_hidden.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, encoder_hidden.transpose(1, 2))  # [batch_size, out_len, in_len]
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, encoder_hidden)  # [batch_size, out_len, hidden_size]

        # concat -> (batch, out_len, 2*dim+embed_dim)
        # try:
        #     combined = torch.cat((mix, output, content_embedding), dim=2)
        #     output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size + 384))).view(batch_size, -1,
        #                                                                                         hidden_size)
        # except RuntimeError:
        #     print(mix.size())
        #     print(output.size())
        #     print(content_embedding.size())
        #     combined = torch.cat((mix, output))
        #     output = torch.tanh(self.linear_out2(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        # output -> (batch, out_len, dim)


        return mix, attn
