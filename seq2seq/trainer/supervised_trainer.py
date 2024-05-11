from __future__ import division
import logging
import os
import random
import time
import math
import sys


import torch
import torch.nn as nn
from torch import optim
from sentence_transformers import SentenceTransformer

import seq2seq
from seq2seq.evaluator import Evaluator, Predictor
from seq2seq.loss import NLLLoss, MSELoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint

sys.path.insert(0, '.')
sys.path.insert(0, '..')


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """

    def __init__(self, pad_id, expt_dir='experiment', loss=MSELoss(), batch_size=64,
                 random_seed=None, model_name='rnn', device='cpu', use_sbert=True, use_sbert_seq=True,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        self.model_name = model_name
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.batch_size = batch_size
        self.use_sbert = use_sbert
        self.use_sbert_seq = use_sbert_seq
        self.evaluator = Evaluator(pad_id=pad_id, use_sbert=use_sbert, use_sbert_seq=use_sbert_seq,
                                   model_name=model_name, loss=self.loss, batch_size=batch_size, device=device)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.device = device

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, content, input_lengths, target_variable, model, teacher_forcing_ratio):
        loss = self.loss
        # Forward propagation

        self.optimizer.optimizer.zero_grad()

        if self.model_name == 'rnn':
            # RNN-based seq2seq model
            decoder_outputs, decoder_hidden, other = model(input_variable, content, input_lengths, target_variable,
                                                           self.use_sbert, self.use_sbert_seq, teacher_forcing_ratio)

        elif self.model_name == 'transformer':
            # Transformer model
            decoder_outputs, cos_sim1, cos_sim2 = model(input_variable, content, target_variable,
                                                        self.use_sbert, self.use_sbert_seq, teacher_forcing_ratio)
            if self.use_sbert_seq:
                cos_list1 = cos_sim1.permute(1, 0).tolist()
                cos_list2 = cos_sim2.permute(1, 0).tolist()
                with open('figures/cos1.csv', 'a') as f1:
                    for i in cos_list1:
                        lst = [str(j) for j in i]
                        f1.write(','.join(lst)+'\r\n')
                with open('figures/cos2.csv', 'a') as f1:
                    for m in cos_list2:
                        lst = [str(n) for n in m]
                        f1.write(','.join(lst) + '\r\n')

        elif self.model_name == 'cnn':
            # Conv seq2seq model
            decoder_outputs, other = model(input_variable, content, target_variable,
                                           self.use_sbert, self.use_sbert_seq, teacher_forcing_ratio)

        elif self.model_name == 'ml':
            # Conv seq2seq model
            decoder_outputs, other = model(input_variable, content, target_variable,
                                           self.use_sbert, self.use_sbert_seq, teacher_forcing_ratio)

        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step].unsqueeze(1))
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, batch_size, n_epochs, start_epoch, start_step,
                       dev_data=None, test_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch
        epoch_log_loss_total = 0  # Reset every epoch

        batch_iterator = data
        steps_per_epoch = batch_iterator.getlen() // batch_size
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.process

            model.train(True)
            # for batch in batch_generator:
            for x, y, length, abs in batch_generator():

                step += 1
                step_elapsed += 1

                input_variables = torch.FloatTensor(x).to(self.device)
                target_variables = torch.FloatTensor(y).to(self.device)
                input_lengths = torch.LongTensor(length).to(self.device)

                loss = self._train_batch(input_variables, abs, input_lengths.tolist(), target_variables, model,
                                         teacher_forcing_ratio)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

                # # Checkpoint
                # if step % self.checkpoint_every == 0 or step == total_steps:
                #     Checkpoint(model=model,
                #                optimizer=self.optimizer,
                #                epoch=epoch, step=step,
                #                input_vocab=data.fields[seq2seq.src_field_name].vocab,
                #                output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "\nFinished epoch %d: Train %s: %.4f" % \
                      (epoch, self.loss.name, epoch_loss_avg)

            predictor = Predictor(model, self.model_name, self.device)
            if dev_data is not None:
                dev_loss = self.evaluator.evaluate(model, dev_data)
                log_msg += ", Dev %s: %.4f" % (self.loss.name, dev_loss)
                if epoch == start_epoch:
                    best_loss = dev_loss
                    best_epoch = epoch
                    torch.save(model, '{}/{}.ckpt'.format(self.expt_dir, self.model_name))
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_epoch = epoch
                    torch.save(model, '{}/{}.ckpt'.format(self.expt_dir, self.model_name))

            if test_data is not None:
                test_loss = self.evaluator.evaluate(model, test_data, True)
                log_msg += ", Test %s: %.4f" % (self.loss.name, test_loss)

            if epoch == n_epochs:
                train_loss = self.evaluator.evaluate(model, data)
                log_msg += "\nTrain %s from Evaluator: %.4f" % (self.loss.name, train_loss)

            log.info(log_msg)

        best_model = torch.load('{}/{}.ckpt'.format(self.expt_dir, self.model_name))
        best_train_loss = self.evaluator.evaluate(best_model, data)
        best_info = "Epoch %d, Train %s: %.4f" % (best_epoch, self.loss.name, best_train_loss)
        if dev_data is not None:
            best_dev_loss = self.evaluator.evaluate(best_model, dev_data)
            best_info += ", Dev %s: %.4f" % (self.loss.name, best_dev_loss)
        if test_data is not None:
            best_test_loss = self.evaluator.evaluate(model, test_data)
            best_info += ", Test %s: %.4f" % (self.loss.name, best_test_loss)
        print("--------------Best Model:--------------")
        print(best_info)
        print("---------------------------------------")

    def train(self, model, data, num_epochs,
              resume=False, dev_data=None, test_data=None, optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            test_data:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """

        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, self.batch_size, num_epochs,
                            start_epoch, step, dev_data=dev_data, test_data=test_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
