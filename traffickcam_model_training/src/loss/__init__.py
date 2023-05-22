from .triplet_margin_loss import TripletMarginLoss
from .ephn_loss import EPHNLoss
from .batchall_loss import BatchAllLoss

_factory = {
    'triplet_margin_loss': TripletMarginLoss,
    'ephn_loss': EPHNLoss,
    'batchall_loss': BatchAllLoss,
}


def names():
    return sorted(_factory.keys())


def create(name, *args, **kwargs):
    if name not in _factory:
        raise KeyError("Unknown loss:", name)
    return _factory[name](*args, **kwargs)
