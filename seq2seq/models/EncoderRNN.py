import copy
import math
import torch
import torch.nn as nn

from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, seq_len, hidden_size,
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='gru',
                 variable_lengths=False, use_sbert=False, use_sbert_seq=False):
        super(EncoderRNN, self).__init__(vocab_size, seq_len, hidden_size,
                                         input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        # self.embedding = nn.Embedding(vocab_size, hidden_size)
        # if embedding is not None:
        #     self.embedding.weight = nn.Parameter(embedding)
        # self.embedding.weight.requires_grad = update_embedding
        # self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
        # input_size = hidden_size // 2
        self.fc_input = nn.Linear(1, hidden_size)
        self.fc_sbert = nn.Linear(hidden_size + 384, hidden_size)
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        # self.rnn = self.rnn_cell(1, hidden_size, n_layers,
        #                          batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        direction = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_size * direction)
        self.use_sbert = use_sbert
        self.use_sbert_seq = use_sbert_seq
        self.mlp = nn.Linear(384, hidden_size)

    def forward(self, input_var, content, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            content:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        # input_var = self.norm(input_var)
        # seq_len = input_var.size(1)

        input_var = input_var.unsqueeze(2)
        input_var = self.fc_input(input_var)

        # input_var = torch.stack([content for i in range(5)], dim=1)
        # input_var = self.mlp(input_var)

        in_len = input_var.size(1)
        if self.use_sbert_seq:
            input_var = torch.cat([input_var, content[:, :in_len, :]], dim=2)  # [batch_size, seq_len, hidden_size+384]
            input_var = self.fc_sbert(input_var)  # [batch_size, seq_len, hidden_size]

        # print("EncoderRNN  input_var:", input_var[:5], input_var.size())
        if self.variable_lengths:
            input_var = nn.utils.rnn.pack_padded_sequence(input_var, input_lengths, batch_first=True)
        output, hidden = self.rnn(input_var)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # output = self.norm(output)

        # content: [batch_size]

        # print("encoder:", output.size())
        return output, hidden, content
