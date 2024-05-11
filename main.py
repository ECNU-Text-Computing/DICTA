import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TransformerEncoder, TransformerDecoder, Transformer
from seq2seq.models import EncoderCNN, DecoderCNN, Seq2seqCNN
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import DataGenerator
from seq2seq.evaluator import Predictor
from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint

sys.path.insert(0, '.')
sys.path.insert(0, '..')

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


os.environ['CUDA_VISIBLE_DEVICES'] = "3"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path', default='pmc',
                    help='Path to train, dev and test data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory '
                         'has to be provided')
parser.add_argument('--log-level', dest='log_level', default='info',
                    help='Logging level.')
parser.add_argument('--model', action='store', dest='model', default='transformer',
                    help='The name of the employed model.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)


# hyperparameter setting
train_path = "datasets/" + opt.data_path + "/train/data.txt"
dev_path = "datasets/" + opt.data_path + "/dev/data.txt"
test_path = "datasets/" + opt.data_path + "/test/data.txt"
model_name = opt.model
batch_size = 64
train = DataGenerator(input_path=train_path, batch_size=batch_size, shuffle=False)
dev = DataGenerator(input_path=dev_path, batch_size=batch_size, shuffle=False)
test = DataGenerator(input_path=test_path, batch_size=batch_size, shuffle=False)

vocab_size = 10000
loss = Perplexity()
loss.tocuda(DEVICE)
seq2seq = None
optimizer = None

input_size = 128
hidden_size = 128  # 16 → 512
num_head = 2
num_epochs = 30
teacher_forcing_ratio = 0.5  # 0.5 → 1
dropout_rate = 0.1  # 0.2 - 0.5
learning_rate = 0.001  # 0.001 - 0.00001
num_layers = 3  # 2 - 5
kernel_size = 3
use_adamw = False
bidirectional = False
use_attention = True
use_sbert = False
use_sbert_seq = False
if not use_sbert:
    use_sbert_seq = False
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
# sbert_model = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
sbert_model = sbert_model.to(DEVICE)
sbert_seq_model = nn.LSTM(384, 384, 3, batch_first=True, dropout=0.1, bidirectional=False)
sbert_seq_model = sbert_seq_model.to(DEVICE)
rnn_cell = 'gru'
pad_id = -1
in_len = 4 if opt.data_path == 'pubmed' or opt.data_path == 'dblp' else 5
out_len = 5

# print hyperparameters
print('------Hyper-parameters------')
print('model_name: {}, dataset: {}, batch_size: {}, hidden_size: {}, learning_rate: {}, num_epochs: {}, num_layers: {}, '
      'dropout_rate: {}, teacher_forcing_ratio: {}'.format(model_name, opt.data_path, batch_size, hidden_size,
                                                           learning_rate, num_epochs, num_layers,
                                                           dropout_rate, teacher_forcing_ratio))
print(
    'input_size: {}, kernel_size: {}, num_head: {}, use_adamw: {}'.format(input_size, kernel_size, num_head, use_adamw))
print('use_sbert = {}, use_sbert_seq = {}, in_len: {}, out_len: {}'.format(use_sbert, use_sbert_seq, in_len, out_len))
print('----------------------------')

if model_name == 'rnn':
    # RNN-based seq2seq model
    encoder = EncoderRNN(vocab_size, in_len, hidden_size, n_layers=num_layers, bidirectional=bidirectional,
                         rnn_cell=rnn_cell, variable_lengths=True, use_sbert=use_sbert, use_sbert_seq=use_sbert_seq)
    decoder = DecoderRNN(vocab_size, out_len, hidden_size * 2 if bidirectional else hidden_size,
                         dropout_p=dropout_rate, use_attention=use_attention, bidirectional=bidirectional,
                         n_layers=num_layers, rnn_cell=rnn_cell, use_sbert=use_sbert, use_sbert_seq=use_sbert_seq,
                         eos_id=-1, sos_id=-1, device=DEVICE)
    seq2seq = Seq2seq(encoder, decoder, sbert_model, sbert_seq_model, out_len, DEVICE)

elif model_name == 'transformer':
    # Transformer model
    encoder = TransformerEncoder(input_size, num_head, hidden_size, dropout_rate, num_layers, use_sbert, use_sbert_seq)
    decoder = TransformerDecoder(input_size, num_head, hidden_size, dropout_rate, num_layers,
                                 pad_idx=pad_id, seq_len=out_len, use_attention=use_attention,
                                 use_sbert=use_sbert, use_sbert_seq=use_sbert_seq, device=DEVICE)
    seq2seq = Transformer(encoder, decoder, sbert_model, sbert_seq_model, out_len, DEVICE)

elif model_name == 'cnn':
    # Conv seq2seq model
    encoder = EncoderCNN(input_size, hidden_size, num_layers, kernel_size, dropout_rate, use_sbert,
                         use_sbert_seq, DEVICE)
    decoder = DecoderCNN(input_size, hidden_size, num_layers, kernel_size, dropout_rate, pad_id, DEVICE,
                         out_len, use_attention, use_sbert, use_sbert_seq)
    seq2seq = Seq2seqCNN(encoder, decoder, sbert_model, sbert_seq_model, out_len, DEVICE)


seq2seq = seq2seq.to(DEVICE)


# Optimizer and learning rate scheduler can be customized by explicitly constructing the objects
# and pass to the trainer.
if use_adamw:
    optimizer = Optimizer(torch.optim.AdamW(seq2seq.parameters(), lr=learning_rate), max_grad_norm=5)
else:
    optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=learning_rate), max_grad_norm=5)

# scheduler = StepLR(optimizer.optimizer, 1)
# optimizer.set_scheduler(scheduler)

# train
t = SupervisedTrainer(pad_id=vocab_size + 3, loss=loss, batch_size=batch_size,
                      model_name=model_name, device=DEVICE,  use_sbert=use_sbert, use_sbert_seq=use_sbert_seq,
                      checkpoint_every=2000, print_every=500,
                      expt_dir=opt.expt_dir + '/' + opt.data_path)

seq2seq = t.train(seq2seq, train,
                  num_epochs=num_epochs, dev_data=dev, test_data=test,
                  optimizer=optimizer, teacher_forcing_ratio=teacher_forcing_ratio)


# test for user input
predictor = Predictor(seq2seq, model_name, DEVICE)
# print("Test", predictor.predict(test, batch_size, model_name))

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print('The predicted sequence is:', predictor.predict1(seq))
