import torch
from torch.autograd import Variable
import numpy as np
import math

import seq2seq
from seq2seq.loss import Perplexity
from sentence_transformers import SentenceTransformer


class Predictor(object):

    def __init__(self, model, model_name='rnn', device='cpu'):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def get_decoder_features(self, src_seq):
        if src_seq[-1].isdigit():
            content = "There is no abstract for this paper."
        else:
            content = src_seq.pop()

        src_id_seq = torch.FloatTensor([[np.log(1 + float(i)) for i in src_seq]]).view(1, -1)
        src_id_seq = src_id_seq.to(self.device)

        with torch.no_grad():
            if self.model_name == 'rnn':
                out_list, _, other = self.model(src_id_seq, [len(src_seq)])
            elif self.model_name == 'transformer':
                out_list = self.model(src_id_seq, content)
            elif self.model_name == 'cnn':
                out_list, _ = self.model(src_id_seq)

        return out_list

    def predict(self, data):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        """ Evaluate a model on given dataset and return performance.

                Args:
                    model (seq2seq.models): model to evaluate
                    data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

                Returns:
                    loss (float): loss of the given model on the given dataset
                """

        loss = Perplexity()
        loss.reset()

        batch_iterator = data

        with torch.no_grad():
            for x, y, length, content in batch_iterator.process():
                input_variables = torch.FloatTensor(x).to(self.device)
                target_variables = torch.FloatTensor(y).to(self.device)
                input_lengths = torch.FloatTensor(length).to(self.device)

                if self.model_name == 'rnn':
                    # RNN-based seq2seq model
                    decoder_outputs, decoder_hidden, other = self.model(input_variables, content,
                                                                        input_lengths.tolist())

                elif self.model_name == 'transformer':
                    # Transformer model
                    decoder_outputs = self.model(input_variables)

                elif self.model_name == 'cnn':
                    # Conv seq2seq model
                    decoder_outputs, other = self.model(input_variables)

                # Evaluation
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target.unsqueeze(1))

        return loss.get_loss()

    def predict1(self, src_seq):
        out_list = self.get_decoder_features(src_seq)

        tgt_tensor = torch.stack(out_list, dim=1)
        tgt_list = tgt_tensor[0].cpu().numpy().tolist()
        # print('tgt_list:', tgt_list)
        tgt_seq = " ".join([str(round(math.exp(i[0]) - 1)) for i in tgt_list])
        return tgt_seq

    def predict_n(self, src_seq, n=1):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            result.append(tgt_id_seq)

        return result
