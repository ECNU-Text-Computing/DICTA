import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    # def __init__(self, encoder, decoder, decode_function=F.log_softmax):
    def __init__(self, encoder, decoder, sbert_model, sbert_seq_model, out_len=5, device='cpu',
                 decode_function=torch.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out_len = out_len
        self.decode_function = decode_function
        self.sbert_seq_model = sbert_seq_model
        self.sbert_model = sbert_model
        self.device = device

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, content, input_lengths=None, target_variable=None,
                use_sbert=True, use_sbert_seq=True, teacher_forcing_ratio=0):

        in_len = input_variable.size(1)
        out_len = target_variable.size(1) if target_variable is not None else self.out_len
        seq_len = in_len + out_len
        if use_sbert_seq:
            content = self.sbert_model.encode(content)  # [batch_size, embed_dim=384]
            content = torch.from_numpy(content).to(self.device)  # [batch_size, 384]
            content = content.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, 384]
            content, _ = self.sbert_seq_model(content)  # [batch_size, seq_len, 384]
        elif use_sbert:
            content = self.sbert_model.encode(content)  # [batch_size, embed_dim=384]
            content = torch.from_numpy(content).to(self.device)  # [batch_size, embed_dim=384]

        input_variable = input_variable.to(torch.float32)
        encoder_outputs, encoder_hidden, content_embedding = self.encoder(input_variable, content, input_lengths)
        result = self.decoder(content_embedding=content_embedding,
                              inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
