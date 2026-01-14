from contextlib import nullcontext
import torch

class MixedPrecisionManager:
    def __init__(self, fp16):
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def context(self):
        if self.fp16:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def backward(self, loss):
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        if self.fp16:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
