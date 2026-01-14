import torch


class L0:
    """L0
    """

    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()


class FLOPS:
    """Minimize FLOPs to Learn Efficient Sparse Representations
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)



class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t


def init_regularizer(reg):
    if reg == "L0":
        return L0()
    elif reg == "FLOPS":
        return FLOPS()
    else:
        raise NotImplementedError("provide valid regularizer!")
