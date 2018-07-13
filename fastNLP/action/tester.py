import numpy as np
import torch

from fastNLP.action.action import RandomSampler, Batchifier


class BaseTester(object):
    """docstring for Tester"""

    def __init__(self, action, test_args):
        """
        :param test_args: named tuple
        """
        super(BaseTester, self).__init__()
        self.action = action
        self.validate_in_training = test_args["validate_in_training"]
        self.save_dev_data = None
        self.save_output = test_args["save_output"]
        self.output = None
        self.save_loss = test_args["save_loss"]
        self.mean_loss = None
        self.batch_size = test_args["batch_size"]
        self.pickle_path = test_args["pickle_path"]
        self.iterator = None

        self.model = None
        self.eval_history = []

    def test(self, network):
        # print("--------------testing----------------")
        self.model = network

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)

        ret = self.action.prepare_input(self.pickle_path + "data_train.pkl")
        dev_data = ret[0]

        self.iterator = iter(Batchifier(RandomSampler(dev_data), self.batch_size, drop_last=True))

        batch_output = list()
        num_iter = len(dev_data) // self.batch_size

        for step in range(num_iter):
            batch_x, batch_y = self.action.batchify(dev_data, self.iterator)

            prediction = self.action.data_forward(network, batch_x)
            eval_results = self.evaluate(prediction, batch_y)

            if self.save_output:
                batch_output.append(prediction)
            if self.save_loss:
                self.eval_history.append(eval_results)

    def evaluate(self, predict, truth):
        raise NotImplementedError

    @property
    def matrices(self):
        raise NotImplementedError

    def mode(self, model, test=True):
        """To do: combine this function with Trainer ?? """
        if test:
            model.eval()
        else:
            model.train()
        self.eval_history.clear()


class POSTester(BaseTester):
    """
    Tester for sequence labeling.
    """

    def __init__(self, action, test_args):
        super(POSTester, self).__init__(action, test_args)
        self.max_len = None
        self.mask = None
        self.batch_result = None
        self.loss_func = None

    def evaluate(self, predict, truth):
        truth = torch.Tensor(truth)
        loss, prediction = self.model.loss(predict, truth, self.action.mask, self.action.batch_size,
                                           self.action.max_len)
        return loss.data

    def matrices(self):
        return np.mean(self.eval_history)

    def get_loss(self, predict, truth):
        """
        Compute loss given prediction and ground truth.
        :param predict: prediction label vector, [batch_size, num_classes]
        :param truth: ground truth label vector, [batch_size, max_len]
        :return: a scalar
        """
        truth = torch.Tensor(truth)
        if self.loss_func is None:
            if hasattr(self.model, "loss"):
                self.loss_func = self.model.loss
            else:
                self.define_loss()
        loss, prediction = self.loss_func(predict, truth, self.mask, self.batch_size, self.max_len)
        # print("loss={:.2f}".format(loss.data))
        return loss

    def define_loss(self):
        self.loss_func = torch.nn.CrossEntropyLoss()
