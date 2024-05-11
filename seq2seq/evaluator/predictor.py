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
        # if torch.cuda.is_available():
        #     self.model = model.cuda()
        # else:
        #     self.model = model.cpu()
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        # self.src_vocab = src_vocab
        # self.tgt_vocab = tgt_vocab

    def get_decoder_features(self, src_seq):
        # print(src_seq[-1].isdigit())
        if src_seq[-1].isdigit():
            content = "There is no abstract for this paper."
        else:
            content = src_seq.pop()

        src_id_seq = torch.FloatTensor([[np.log(1 + float(i)) for i in src_seq]]).view(1, -1)
        # print('input_list:', src_id_seq)
        # print(content)
        # content = self.smodel.encode(content)
        # content = torch.from_numpy(content).to(self.device)
        src_id_seq = src_id_seq.to(self.device)
        # print("src_seq:{}, src_id_seq:{}, len(src_seq):{}".format(src_seq, src_id_seq, len(src_seq)))

        with torch.no_grad():
            if self.model_name == 'rnn':
                out_list, _, other = self.model(src_id_seq, [len(src_seq)])
            elif self.model_name == 'transformer':
                out_list = self.model(src_id_seq, content)
            elif self.model_name == 'cnn':
                out_list, _ = self.model(src_id_seq)

        # pre_list = []
        # for step, step_output in enumerate(softmax_list):
        #     pre_list.append(step_output.contiguous().view(batch_size, -1))

        return out_list

    def predict(self, data, batch_size):
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
        # a = batch_iterator.getlen() // batch_size
        # steps = 0

        with torch.no_grad():
            for x, y, length, content in batch_iterator.process():
                # steps += 1

                # input_variables, input_lengths  = batch_x, len(batch_x)
                # target_variables = batch_y
                input_variables = torch.FloatTensor(x).to(self.device)
                target_variables = torch.FloatTensor(y).to(self.device)
                input_lengths = torch.FloatTensor(length).to(self.device)

                # decoder_outputs, decoder_hidden, other = self.model(input_variables, input_lengths.tolist(),
                #                                                target_variables)

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

                # last_input = input_variables[:, -1]

                # Evaluation
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target.unsqueeze(1))

        # avg_loss, avg_log_loss = loss.get_loss_p()
        # return '%s: %.4f' % (loss.name, avg_loss)
        return loss.get_loss()

    def predict1(self, src_seq):
        out_list = self.get_decoder_features(src_seq)

        tgt_tensor = torch.stack(out_list, dim=1)
        tgt_list = tgt_tensor[0].cpu().numpy().tolist()
        # print('tgt_list:', tgt_list)
        tgt_seq = " ".join([str(round(math.exp(i[0]) - 1)) for i in tgt_list])

        # RNN-based model
        # if self.model_name == 'rnn' or self.model_name == 'cnn':
        #     tgt_tensor = torch.stack(out_list, dim=1)
        #     tgt_list = tgt_tensor[0].cpu().numpy().tolist()
        #     print('tgt_list:', tgt_list)
        #     tgt_seq = " ".join([str(round(math.exp(i[0])-1, 1)) for i in tgt_list])

        # Transformer model
        # elif self.model_name == 'transformer':
        #     tgt_tensor = out_list.squeeze()
        #     tgt_list = tgt_tensor.cpu().numpy().tolist()
        #     tgt_seq = " ".join([str(round(i)) for i in tgt_list])

        # CNN model
        # elif self.model_name == 'transformer':
        #     tgt_tensor = out_list.squeeze()
        #     tgt_list = tgt_tensor.cpu().numpy().tolist()
        #     print('tgt_list:', tgt_list)
        #     tgt_seq = " ".join([str(round(math.exp(i)-1, 1)) for i in tgt_list])

        # print("tgt_tensor: {}, tgt_list: {}, tgt_seq: {}".format(tgt_tensor, tgt_list, tgt_seq))

        # length = other['length'][0]
        # print(other['sequence'][:5], len(other['sequence']))
        #
        # tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        # # tgt_id_seq = [other['sequence'][di][0] for di in range(length)]
        # tgt_id_list = torch.stack(tgt_id_seq).cpu().numpy().tolist()
        # # print("tgt_id_seq:", tgt_id_seq.data[0])
        # # tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        # tgt_seq = " ".join([str(i) for i in tgt_id_list])
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
            # tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            # result.append(tgt_seq)
            result.append(tgt_id_seq)

        return result
