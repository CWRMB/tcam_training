from pytorch_metric_learning import miners, losses

from .base_loss import BaseLoss


class TripletMarginLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.margin = kwargs['margin']

        """ Computes triplet loss using all triplets that violate the margin """
        self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="all")
        self.loss = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        triplets = self.miner(embeddings, labels)
        loss = self.loss(embeddings, labels, triplets)

        return loss
