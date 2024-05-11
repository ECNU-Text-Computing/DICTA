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
        # print("Load data from {}.".format(self.input_path))

        # 读取输入数据。
        with open(self.input_path, 'r') as fp:
            input_data = fp.readlines()

        # 是否将数据随机打乱？
        if self.shuffle:
            random.shuffle(input_data)

            # 按照batch将数据输出给深度学习模型。
            # 注意，因为下面使用了yield而非return，所以这个函数是一个生成器。具体的使用见深度学习Base_Model的train部分。

        batch_size = self.batch_size
        for i in range(0, len(input_data), batch_size):
            # print(i)
            j = 0
            # 每次循环开始时，都先清空batch_input。
            batch_input, batch_output, batch_len, batch_abs = [], [], [], []
            # 读取输入数据的下一个batch。
            for line in input_data[i: i + batch_size]:
                j += 1
                new_input, new_output = [], []
                if abs(len(line.strip().split('\t')) - 3) < 0.000001:
                    input_list, output_list, abstract = line.strip().split('\t')
                    # ['1 2 3 4 5', '2 3 4 5 6', 'abstract']
                elif abs(len(line.strip().split('\t')) - 2) < 0.000001:
                    # print('batch: ' + str(i))
                    # print('num: ' + str(j))
                    # print('data: ' + line)
                    input_list, output_list = line.strip().split('\t')
                    abstract = 'There is no abstract for this paper'
                else:
                    continue

                len_input = len(input_list.strip().split())
                # 将字符的列表转化为id的列表。
                for p in input_list.strip().split():
                    new_input.append(np.log(1 + float(p)))
                    # new_input.append(float(p))
                for q in output_list.strip().split():
                    new_output.append(np.log(1 + float(q)))
                    # new_output.append(float(q))
                    # if i < len_input :
                    #     a = float(word_list[i])
                    #     a = np.log(1 + a)
                    #     new_input.append(a)
                    #     # new_input.append(float(word_list[i]))
                    # else:
                    #     b = float(word_list[i])
                    #     b = np.log(1 + b)
                    #     new_output.append(b)
                    # new_output.append(float(word_list[i]))
                # 将转化后的id列表append到batch_input中。
                # 因为要使用toch的pad函数，所以此处就要把每个id列表转化为torch的Tensor格式。
                batch_input.append(torch.FloatTensor(new_input))
                batch_output.append(torch.FloatTensor(new_output))
                batch_len.append(len_input)
                batch_abs.append(abstract)

            # batch_x = pad_sequences(new_batch, maxlen=None)
            # 将不等长的数据进行pad操作。
            # [[1, 2], [1, 2, 3]] ==> [[1, 2, 0], [1, 2, 3]]
            batch_x = pad_sequence(batch_input, batch_first=True).detach().numpy()
            batch_y = pad_sequence(batch_output, batch_first=True).detach().numpy()

            yield batch_x, batch_y, batch_len, batch_abs
