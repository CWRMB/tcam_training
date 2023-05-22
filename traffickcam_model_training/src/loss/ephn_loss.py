from pytorch_metric_learning import miners, losses

from .base_loss import BaseLoss


class EPHNLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        """ This combo replicates Hong's EPHN loss using PTML """
        self.miner = miners.BatchEasyHardMiner(pos_strategy='easy', neg_strategy='hard')
        self.loss = losses.NTXentLoss(temperature=1.0)

    def forward(self, embeddings, labels):
        triplets = self.miner(embeddings, labels)
        loss = self.loss(embeddings, labels, triplets)

        return loss
