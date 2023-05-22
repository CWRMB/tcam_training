import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .base_visualization import BaseVisualization


class ConfusionMatrix(BaseVisualization):
    def __init__(self, logger):
        super().__init__(logger)

    @staticmethod
    def build_matrix(queries, predictions):
        matrix = confusion_matrix(queries, predictions)
        return pd.DataFrame(matrix)

    def log(self, outputs: dict):
        for phase in ['training', 'validation']:
            plt.clf()
            plt.figure(figsize=(10, 7))
            plt.title(phase)
            print(outputs[phase]['labels'].shape, outputs[phase]['knn_labels'].shape)
            raise NotImplementedError
            # matrix = self.build_matrix(outputs[phase]['labels'],
            #                            np.transpose(outputs[phase]['knn_labels'][0]))
            # sns.heatmap(matrix, annot=True)
            # plt.savefig('confusion_matrix.png')
            # self.logger.log_image('confusion_matrix', 'confusion_matrix.png')
