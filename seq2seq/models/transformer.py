import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


import copy
import torch


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, sbert_model, sbert_seq_model, out_len, device='cuda'):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out_len = out_len
        self.sbert_model = sbert_model
        self.sbert_seq_model = sbert_seq_model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.smodel = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    # def forward(self, input_variable, input_lengths=None, target_variable=None,
    #             teacher_forcing_ratio=0):
    def forward(self, src, content, tgt=None, use_sbert=True, use_sbert_seq=True, teacher_forcing_ratio=0,
                src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, if_draw=False):

        in_len = src.size(1)
        out_len = tgt.size(1) if tgt is not None else self.out_len
        seq_len = in_len + out_len
        if use_sbert:
            content = self.sbert_model.encode(content)  # [batch_size, embed_dim=384]
            content = torch.from_numpy(content).to(self.device)  # [batch_size, embed_dim=384]
            if use_sbert_seq:
                content = content.unsqueeze(-2).repeat(1, seq_len, 1)  # [batch_size, seq_len, 384]
                # print(content.device, next(self.sbert_seq_model.parameters()).device)
                content, _ = self.sbert_seq_model(content)  # [batch_size, seq_len, 384]

        src = src.unsqueeze(2).permute(1, 0, 2)
        if isinstance(tgt, list):
            print(tgt[:5], len(tgt))
        if tgt is not None:
            tgt = tgt.unsqueeze(2).permute(1, 0, 2)
        encoder_outputs, content_embedding = self.encoder(src, content,
                                                          mask=src_mask,
                                                          src_key_padding_mask=src_key_padding_mask)
        # if use_sbert_seq:
        #     cos_list1 = []
        #     cos_list2 = []
        #     for i in range(1, content_embedding.size(1)):
        #         cos_list1.append(F.cosine_similarity(content_embedding[:, 0, :], content_embedding[:, i, :]))
        #         cos_list2.append(F.cosine_similarity(content_embedding[:, i-1, :], content_embedding[:, i, :]))
        #     cos_sim1 = torch.stack(cos_list1)
        #     cos_sim2 = torch.stack(cos_list2)

        result = self.decoder(content_embedding=content_embedding,
                              memory=encoder_outputs,
                              tgt=tgt,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return result
        # return result, cos_sim1, cos_sim2
