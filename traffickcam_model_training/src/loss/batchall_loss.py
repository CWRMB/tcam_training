from pytorch_metric_learning import miners, losses

from .base_loss import BaseLoss


class BatchAllLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.margin = kwargs['margin']

        """ Computes triplet loss using all triplets """
        self.miner = miners.TripletMarginMiner(margin=1.0, type_of_triplets="all")
        self.loss = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        triplets = self.miner(embeddings, labels)
        loss = self.loss(embeddings, labels, triplets)

        return loss
