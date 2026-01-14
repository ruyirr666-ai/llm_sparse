import torch

class custom_save:
    def __init__(self, config):
        self.config = config

    def save_checkpoint(self, state, path):
        torch.save(state, path)
