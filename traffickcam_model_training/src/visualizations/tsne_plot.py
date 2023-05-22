import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from .base_visualization import BaseVisualization


class TSNEPlot(BaseVisualization):
    def __init__(self, logger):
        super().__init__(logger)

    def map_features(self, outputs, labels, phase):
        # create array of column for each feature output
        feat_cols = ['feature' + str(i) for i in range(outputs.shape[1])]

        # make dataframe of outputs -> labels
        df = pd.DataFrame(outputs, columns=feat_cols)
        df['y'] = labels
        df['labels'] = df['y'].apply(lambda i: str(i))

        # creates an array of random indices from size of outputs
        np.random.seed(42)
        rand_perm = np.random.permutation(df.shape[0])

        num_examples = 10000

        df_subset = df.loc[rand_perm[:num_examples], :].copy()
        data_subset = df_subset[feat_cols].values

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]

        plt.clf()
        plt.figure(figsize=(16, 10))
        plt.scatter(
            x=df_subset["tsne-2d-one"],
            y=df_subset["tsne-2d-two"],
            c=df_subset["y"],
            s=4
        )
        plt.axis('off')
        plt.title("{} Embedding Space".format(phase.capitalize()))
        plt.savefig('tsne.png')
        self.logger.log_image('{} t-SNE'.format(phase), 'tsne.png')

    def log(self, outputs: dict):
        for phase in ['training', 'validation']:
            embeddings = np.concatenate((outputs[phase]['embeddings'], outputs['gallery']['embeddings']), axis=0)
            labels = np.concatenate((outputs[phase]['labels'], outputs['gallery']['labels']), axis=0)
            self.map_features(embeddings, labels, phase)
