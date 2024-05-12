from __future__ import print_function
import math
import torch.nn as nn
import numpy as np
import torch


class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # Accumulated loss
        self.acc_loss = 0
        # Normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def tocuda(self, device):
        self.criterion = self.criterion.to(device)

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()


class MSELoss(Loss):
    _NAME = "Avg MSELoss"

    def __init__(self, reduction='sum'):
        self.reduction = reduction
        super(MSELoss, self).__init__(self._NAME, nn.MSELoss(reduction='sum'))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # Total loss for all batches
        loss = self.acc_loss.data.item()
        # print(loss)
        # if self.reduction == 'mean':
        #     # average loss per batch
        #     loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += outputs.size(0)


class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        print(weight)
        super(NLLLoss, self).__init__(
            self._NAME,
            nn.NLLLoss(weight=weight, size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data.item()
        # print(loss)
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1


class Perplexity(MSELoss):
    _NAME = "RMSE Loss"
    _MAX_EXP = 100

    def __init__(self):
        # super(Perplexity, self).__init__(weight=weight, mask=mask, size_average=False)
        super(Perplexity, self).__init__()

    def eval_batch(self, outputs, target):
        # loss = self.criterion(outputs, target)
        # loss = math.sqrt(loss)
        self.acc_loss += torch.sqrt(self.criterion(outputs, target))
        # self.norm_term += batch_size
        self.norm_term += outputs.size(0)
        # self.norm_term += 1
        # print('outputs, target:', outputs, target)
        # print('acc_loss:', self.acc_loss)
        # self.norm_term += target.sum()
        # 手动计算一条数据，看Loss的算法
        # 加个logmale
        # epoches调到50
        # 输入log(1 + x)

    def get_loss(self):
        mse = super(Perplexity, self).get_loss()
        # mse /= self.norm_term.item()
        mse /= self.norm_term
        # if mse > Perplexity._MAX_EXP:
        #     print("WARNING: Loss exceeded maximum value, capping to e^100")
        #     return math.exp(Perplexity._MAX_EXP)
        return mse

    @property
    def MAX_EXP(self):
        return self._MAX_EXP


class NLLPerplexity(NLLLoss):
    """ Language model perplexity loss.

    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None):
        super(NLLPerplexity, self).__init__(weight=weight, mask=mask, size_average=False)

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        nll = super(NLLPerplexity, self).get_loss()
        nll /= self.norm_term.item()
        if nll > Perplexity.MAX_EXP:
            print("WARNING: Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity.MAX_EXP)
        return math.exp(nll)
