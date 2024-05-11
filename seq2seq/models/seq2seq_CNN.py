import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class Seq2seqCNN(nn.Module):
    def __init__(self, encoder, decoder, sbert_model, sbert_seq_model, out_len, device='cpu'):
        super(Seq2seqCNN, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.out_len = out_len
        self.device = device
        self.sbert_seq_model = sbert_seq_model
        self.sbert_model = sbert_model

    def forward(self, src, content, tgt=None, use_sbert=True, use_sbert_seq=True, teacher_forcing_ratio=0):
        # src = [batch size, src sent len]
        # tgt = [batch size, trg sent len]

        in_len = src.size(1)
        out_len = tgt.size(1) if tgt is not None else self.out_len
        seq_len = in_len + out_len
        if use_sbert_seq:
            content = self.sbert_model.encode(content)  # [batch_size, embed_dim=384]
            content = torch.from_numpy(content).to(self.device)  # [batch_size, 384]
            content = content.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, 384]
            content, _ = self.sbert_seq_model(content)  # [batch_size, seq_len, 384]
        elif use_sbert:
            content = self.sbert_model.encode(content)  # [batch_size, embed_dim=384]
            content = torch.from_numpy(content).to(self.device)  # [batch_size, embed_dim=384]

        src = src.unsqueeze(2)  # [batch size, src sent len, 1]
        if tgt is not None:
            tgt = tgt.unsqueeze(2)  # [batch size, src sent len, 1]

        # calculate z^u (encoder_conved) and e (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus positional embeddings
        encoder_conved, encoder_combined, content_embedding = self.encoder(src, content)

        # encoder_conved = [batch size, src sent len, emb dim]
        # encoder_combined = [batch size, src sent len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for each word in the trg sentence

        # output, attention = self.decoder(tgt, encoder_conved, encoder_combined)
        # tgt = torch.tensor(tgt)
        output, attention = self.decoder(content_embedding, encoder_conved, encoder_combined, tgt, teacher_forcing_ratio)

        # output = [batch size, trg sent len, output dim]
        # attention = [batch size, trg sent len, src sent len]

        return output, attention