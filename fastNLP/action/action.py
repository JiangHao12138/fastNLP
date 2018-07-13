import _pickle

import numpy as np
import torch

from fastNLP.modules.utils import seq_mask


class Action(object):
    """Action provides common operations shared by Trainer and Tester, which makes the both
        only a framework focusing on training and testing logic.

        Subclasses must implement the following abstract methods:
        - prepare_input
        - mode
        - define_optimizer
        - data_forward
        - grad_backward
        - get_loss
    """

    def __init__(self):
        super(Action, self).__init__()
        self.iterator = None

    @staticmethod
    def prepare_input(*data_path):
        """Load any number of pickle files.
        :param data_path: str, the path to pickle file
        :return data_list: list, containing pieces of data in the same order as arguments
        """
        data_list = []
        for data_file in data_path:
            data_list.append(_pickle.load(open(data_file, "rb")))
        return data_list

    @staticmethod
    def mode(network, test=False):
        """
        Tell the network to be trained or not.
        :param test: bool
        """
        raise NotImplementedError

    @staticmethod
    def data_forward(network, x):
        """
        Forward pass of the data.
        :param network: a model
        :param x: input feature matrix and label vector
        :return: output by the models

        For PyTorch, just do "network(*x)"
        """
        raise NotImplementedError

    @staticmethod
    def batchify(data, iterator):
        """
        1. Perform batching from data and produce a batch of training data.
        2. Add padding.
        :param iterator: object that has __next__ method, an iterator of data indices.
        :param data: list. Each entry is a sample, which is also a list of features and label(s).
            E.g.
                [
                    [[word_11, word_12, word_13], [label_11. label_12]],  # sample 1
                    [[word_21, word_22, word_23], [label_21. label_22]],  # sample 2
                    ...
                ]
        :return batch_x: list. Each entry is a list of features of a sample. [batch_size, max_len]
                 batch_y: list. Each entry is a list of labels of a sample.  [batch_size, num_labels]
        """
        indices = next(iterator)
        batch = [data[idx] for idx in indices]
        batch_x = [sample[0] for sample in batch]
        batch_y = [sample[1] for sample in batch]
        batch_x = Action.pad(batch_x)
        return batch_x, batch_y

    @staticmethod
    def pad(batch, fill=0):
        """
        Pad a batch of samples to maximum length.
        :param batch: list of list
        :param fill: word index to pad, default 0.
        :return: a padded batch
        """
        max_length = max([len(x) for x in batch])
        for idx, sample in enumerate(batch):
            if len(sample) < max_length:
                batch[idx] = sample + [fill * (max_length - len(sample))]
        return batch


class POSAction(Action):
    """
    Common actions shared by POSTrainer and POSTester.
    """

    def __init__(self):
        super(POSAction, self).__init__()
        self.batch_size = None
        self.max_len = None
        self.mask = None

    @staticmethod
    def mode(network, test=False):
        if test:
            network.eval()
        else:
            network.train()

    def data_forward(self, network, x):
        """
        :param network: the PyTorch model
        :param x: list of list, [batch_size, max_len]
        :return y: [batch_size, num_classes]
        """
        seq_len = [len(seq) for seq in x]
        x = torch.Tensor(x).long()
        self.batch_size = x.size(0)
        self.max_len = x.size(1)
        self.mask = seq_mask(seq_len, self.max_len)
        y = network(x)
        return y


class BaseSampler(object):
    """
        Base class for all samplers.
    """

    def __init__(self, data_set):
        self.data_set_length = len(data_set)

    def __len__(self):
        return self.data_set_length

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(BaseSampler):
    """
    Sample data in the original order.
    """

    def __init__(self, data_set):
        super(SequentialSampler, self).__init__(data_set)

    def __iter__(self):
        return iter(range(self.data_set_length))


class RandomSampler(BaseSampler):
    """
    Sample data in random permutation order.
    """

    def __init__(self, data_set):
        super(RandomSampler, self).__init__(data_set)

    def __iter__(self):
        return iter(np.random.permutation(self.data_set_length))


class Batchifier(object):
    """
    Wrap random or sequential sampler to generate a mini-batch.
    """

    def __init__(self, sampler, batch_size, drop_last=True):
        super(Batchifier, self).__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) < self.batch_size and self.drop_last is False:
            yield batch
