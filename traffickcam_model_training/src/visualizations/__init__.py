from .tsne_plot import TSNEPlot
from .confusion_matrix import ConfusionMatrix
from .knn_images import KNNImages

_factory = {
    'tsne': TSNEPlot,
    'confusion_matrix': ConfusionMatrix,
    'knn_images': KNNImages
}


def names():
    return sorted(_factory.keys())


def create(name, *args, **kwargs):
    if name not in _factory:
        raise KeyError("Unknown loss:", name)
    return _factory[name](*args, **kwargs)
