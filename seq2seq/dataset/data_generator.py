#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataGenerator
======
A class for something.
"""

import os
import random
import sys

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, '.')
sys.path.insert(0, '..')


class DataGenerator(object):
    def __init__(self, input_path, batch_size, shuffle):
        self.input_path = input_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def getlen(self):
        with open(self.input_path, 'r') as fp:
            input_data = fp.readlines()

        len_input = len(input_data)

        return len_input

    def seqlen(self):
        with open(self.input_path, 'r') as fp:
            line = fp.readline()

        input_len = len(line.strip().split()) // 2
        output_len = len(line.strip().split()) // 2

        return input_len, output_len

    def process(self):
        # Print("Load data from {}.".format(self.input_path))

        # Read original data
        with open(self.input_path, 'r') as fp:
            input_data = fp.readlines()

        # shuffle data
        if self.shuffle:
            random.shuffle(input_data)

        # Output the data to the deep learning model in batches.
        # Note that because "yield" is used instead of "return" below, this function is a generator.
        batch_size = self.batch_size
        for i in range(0, len(input_data), batch_size):
            # print(i)
            j = 0
            # Empty batch_input at the beginning of each loop.
            batch_input, batch_output, batch_len, batch_abs = [], [], [], []
            # Read the next batch of input data.
            for line in input_data[i: i + batch_size]:
                j += 1
                new_input, new_output = [], []
                if abs(len(line.strip().split('\t')) - 3) < 0.000001:
                    input_list, output_list, abstract = line.strip().split('\t')
                    # ['1 2 3 4 5', '2 3 4 5 6', 'abstract']
                elif abs(len(line.strip().split('\t')) - 2) < 0.000001:
                    input_list, output_list = line.strip().split('\t')
                    abstract = 'There is no abstract for this paper'
                else:
                    continue

                len_input = len(input_list.strip().split())
                # Convert lists of strings into lists of numbers, and scale them.
                for p in input_list.strip().split():
                    new_input.append(np.log(1 + float(p)))
                for q in output_list.strip().split():
                    new_output.append(np.log(1 + float(q)))
                # Append the converted citation count list to batch_input, and convert them into torch Tensors
                batch_input.append(torch.FloatTensor(new_input))
                batch_output.append(torch.FloatTensor(new_output))
                batch_len.append(len_input)
                batch_abs.append(abstract)

            # Perform padding on the uneven-length data.
            # [[1, 2], [1, 2, 3]] ==> [[1, 2, 0], [1, 2, 3]]
            batch_x = pad_sequence(batch_input, batch_first=True).detach().numpy()
            batch_y = pad_sequence(batch_output, batch_first=True).detach().numpy()

            yield batch_x, batch_y, batch_len, batch_abs
