import torch

from fastNLP.action.action import RandomSampler, Batchifier
from fastNLP.action.tester import POSTester


class BaseTrainer(object):
    """Base trainer for all trainers.
        Trainer receives a model and data, and then performs training.

        Subclasses must implement the following abstract methods:
        - define_optimizer
        - data_forward
        - grad_backward
        - get_loss
        - define_loss
    """

    def __init__(self, action, train_args):
        """
        :param train_args: dict of (key, value)

        The base trainer requires the following keys:
        - epochs: int, the number of epochs in training
        - validate: bool, whether or not to validate on dev set
        - batch_size: int
        - pickle_path: str, the path to pickle files for pre-processing
        """
        super(BaseTrainer, self).__init__()
        self.action = action
        self.n_epochs = train_args["epochs"]
        self.validate = train_args["validate"]
        self.batch_size = train_args["batch_size"]
        self.pickle_path = train_args["pickle_path"]
        self.model = None
        self.iterator = None
        self.loss_func = None
        self.optimizer = None

    def define_optimizer(self):
        """
        Define framework-specific optimizer specified by the models.
        """
        raise NotImplementedError

    def get_loss(self, predict, truth):
        """
        Compute loss given prediction and ground truth.
        :param predict: prediction label vector
        :param truth: ground truth label vector
        :return: a scalar
        """
        if self.loss_func is None:
            if hasattr(self.model, "loss"):
                self.loss_func = self.model.loss
            else:
                self.define_loss()
        return self.loss_func(predict, truth)

    def define_loss(self):
        """
            Assign an instance of loss function to self.loss_func
            E.g. self.loss_func = nn.CrossEntropyLoss()
        """
        raise NotImplementedError

    def grad_backward(self, loss):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def train(self, network):
        """General Training Steps
        :param network: a model

        The method is framework independent.
        Work by calling the following methods:
            - define_optimizer
            - define_loss
            - get_loss
            - grad_backward
            - update
        Subclasses must implement these methods with a specific framework.
        """
        # prepare model and data
        self.model = network

        ret = self.action.prepare_input(self.pickle_path + "data_train.pkl",
                                        self.pickle_path + "data_train.pkl")
        data_train = ret[0]
        data_dev = ret[1]

        # define tester over dev data
        valid_args = {"save_output": True, "validate_in_training": True, "save_dev_input": True,
                      "save_loss": True, "batch_size": self.batch_size, "pickle_path": self.pickle_path}
        validator = POSTester(self.action, valid_args)

        # main training epochs
        iterations = len(data_train) // self.batch_size
        for epoch in range(self.n_epochs):

            # turn on network training mode; define optimizer; prepare batch iterator
            self.action.mode(network, test=False)
            self.define_optimizer()
            self.iterator = iter(Batchifier(RandomSampler(data_train), self.batch_size, drop_last=True))

            # training iterations in one epoch
            for step in range(iterations):
                batch_x, batch_y = self.action.batchify(data_train, self.iterator)

                prediction = self.action.data_forward(network, batch_x)

                loss = self.get_loss(prediction, batch_y)
                self.grad_backward(loss)
                self.update()

            if self.validate:
                if data_dev is None:
                    raise RuntimeError("No validation data provided.")
                validator.test(network)
                print("[epoch {}] dev loss={:.2f}".format(epoch, validator.matrices()))

        # finish training


class POSTrainer(BaseTrainer):
    """
    Trainer for Sequence Modeling

    """

    def __init__(self, action, train_args):
        super(POSTrainer, self).__init__(action, train_args)
        self.vocab_size = train_args["vocab_size"]
        self.num_classes = train_args["num_classes"]
        self.max_len = None
        self.mask = None

    def define_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def grad_backward(self, loss):
        self.model.zero_grad()
        loss.backward()

    def update(self):
        self.optimizer.step()

    def define_loss(self):
        self.loss_func = torch.nn.CrossEntropyLoss()

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
        loss, prediction = self.loss_func(predict, truth, self.action.mask, self.action.batch_size, self.action.max_len)
        # print("loss={:.2f}".format(loss.data))
        return loss


if __name__ == "__name__":
    pass
