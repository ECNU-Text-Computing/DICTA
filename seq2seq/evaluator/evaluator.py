from __future__ import print_function, division

import torch
# import torchtext

import seq2seq
from seq2seq.loss import MSELoss


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, pad_id, use_sbert=True, use_sbert_seq=True, model_name='rnn',
                 loss=MSELoss(), batch_size=64, device='cpu'):
        self.pad = pad_id
        self.use_sbert = use_sbert
        self.use_sbert_seq = use_sbert_seq
        self.model_name = model_name
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

    def evaluate(self, model, data, if_draw=False):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()

        batch_iterator = data

        with torch.no_grad():
            for x, y, length, content in batch_iterator.process():
                # input_variables, input_lengths  = batch_x, len(batch_x)
                # target_variables = batch_y
                input_variables = torch.FloatTensor(x).to(self.device)
                target_variables = torch.FloatTensor(y).to(self.device)
                input_lengths = torch.LongTensor(length).to(self.device)

                if self.model_name == 'rnn':
                    # RNN-based seq2seq model
                    decoder_outputs, decoder_hidden, other = model(input_variables, content, input_lengths.tolist(),
                                                                   use_sbert=self.use_sbert,
                                                                   use_sbert_seq=self.use_sbert_seq)

                elif self.model_name == 'transformer':
                    # Transformer model
                    decoder_outputs, cos_sim1, cos_sim2 = model(input_variables, content, if_draw=if_draw,
                                                                use_sbert=self.use_sbert,
                                                                use_sbert_seq=self.use_sbert_seq)

                elif self.model_name == 'cnn':
                    # Conv seq2seq model
                    decoder_outputs, other = model(input_variables, content,
                                                   use_sbert=self.use_sbert, use_sbert_seq=self.use_sbert_seq)

                # last_input = input_variables[:, -1]

                # Evaluation
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step]
                    loss.eval_batch(step_output.contiguous().view(target_variables.size(0), -1), target.unsqueeze(1))

                if if_draw:
                    cos_list1 = cos_sim1.tolist()
                    cos_list2 = cos_sim2.tolist()
                    with open('cos1.csv', 'a') as f1:
                        for i in cos_list1:
                            f1.write(str(i) + '\r\n')
                    with open('cos2.csv', 'a') as f2:
                        for j in cos_list2:
                            f2.write(str(j) + '\r\n')

        return loss.get_loss()
