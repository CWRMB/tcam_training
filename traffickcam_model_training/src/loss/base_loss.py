from torch.nn.modules.module import Module


class BaseLoss(Module):
    def __init__(self):
        super().__init__()
        self.miner = None
        self.loss = None

    def forward(self):
        raise NotImplementedError
