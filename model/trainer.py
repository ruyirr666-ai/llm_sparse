import os
import time
from model.utils import makedir


class BaseTrainer:
    """
    BaseTrainer
    """

    def __init__(self, model, loss, optimizer, config, train_loader, val_loader=None,
                 scheduler=None, regularizer=None):
        print("initialize trainer...")
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.config = config
        print("train config: \n", self.config)
        assert train_loader is not None, "provide at least train loader"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.validation = True if (self.val_loader is not None) else False
        self.scheduler = scheduler
        self.regularizer = regularizer
        self.fp16 = config["fp16"]
        self.device = config["device"]
        print("total parameters: ", sum([p.nelement() for p in self.model.parameters()]))
        self.model.train()

        self.checkpoint_dir = os.path.join(config["local_path"], config["checkpoint_dir"], config["train_ds"])
        makedir(self.checkpoint_dir)

    def train(self):
        """
        full training logic
        """
        t0 = time.time()
        self.model.train()
        self.train_iterations()
        print("======= TRAINING DONE =======")
        print("train hours: ", (time.time() - t0) / 3600)

    def train_iterations(self):
        raise NotImplementedError


class TrainerIter(BaseTrainer):
    """
    standard class to train a model with a given number of iterations (there is no notion of
    epochs here, for instance when the dataset is large and the model already converged before seeing every example)
    """

    def __init__(self, iterations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # iterations is a tuple with START and END
        self.start_iteration = iterations[0]
        self.nb_iterations = iterations[1]
        self.record_frequency = self.config["record_freq"]
        self.train_iterator = iter(self.train_loader)

    def train_iterations(self):
        raise NotImplementedError
