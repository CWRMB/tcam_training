from pytorch_metric_learning.distances import CosineSimilarity
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import shutil

from .base_visualization import BaseVisualization


class KNNImages(BaseVisualization):
    def __init__(self, logger, count: int):
        super().__init__(logger)
        self.count = count
        self.iter = 0
        for phase in ['training', 'validation']:
            if os.path.exists('./knn_images_{}'.format(phase)):
                shutil.rmtree('./knn_images_{}'.format(phase))
            os.makedirs('./knn_images_{}'.format(phase))

    def log(self, outputs: dict):
        for phase in ['training', 'validation']:
            path_pairs = self.get_path_pairs(outputs[phase], outputs['gallery'])
            self.log_path_pairs(path_pairs, phase)
        self.iter += 1
    
    def get_path_pairs(self, query_outputs, gallery_outputs):
        query_paths, gallery_paths = query_outputs['paths'], gallery_outputs['paths']
        query_labels, gallery_labels = query_outputs['labels'], gallery_outputs['labels']
        path_pairs = []
        similarity_calculator = CosineSimilarity()
        similarities = similarity_calculator(torch.from_numpy(query_outputs['embeddings']), torch.from_numpy(gallery_outputs['embeddings']))
        closest = torch.topk(similarities, k=self.count, dim=1, largest=True)
        for query_idx, ref_indices in enumerate(closest.indices):
            similarity_scores = closest.values[query_idx]
            path_pairs.append(
                [
                    query_paths[query_idx],
                    [
                        (
                            gallery_paths[ref_idx],
                            query_labels[query_idx] == gallery_labels[ref_idx],
                            similarity_scores[j]
                        ) for j, ref_idx in enumerate(ref_indices)
                    ]
                ]
            )
        return path_pairs

    def log_path_pairs(self, path_pairs: list, phase: str):
        for i, (query, references) in enumerate(path_pairs):
            if i % random.randint(5000, 10000) == 0 or i % 10000 == 0:
                title = "{} Nearest Neighbors".format(self.count)
                f, ax = plt.subplots(1, 1+self.count)
                with Image.open(query) as img:
                    ax[0].axis('off')
                    ax[0].imshow(img)
                    ax[0].title.set_text("Query")
                for j, (ref, correct, similarity_score) in enumerate(references):
                    img = np.copy(np.asarray(Image.open(ref)))  # copy for permissions reset
                    ax[j+1].axis('off')
                    ax[j+1].imshow(self.bordered_image(img, correct))
                    ax[j+1].title.set_text("Score: {:0.6f}".format(similarity_score))
                plt.suptitle(title, fontsize=40)
                fig = plt.gcf()
                fig.set_size_inches(4*(1+self.count), 8)
                image_file = "./knn_images_{}/Index{}_{}.png".format(phase, i, self.iter)
                fig.savefig(image_file, dpi=100)
                self.logger.log_image(
                    "{} Nearest Neighbors".format(phase.capitalize()),
                    image_file,
                    image_name="Index {}".format(i)
                )
        self.logger.log_artifact("./knn_images_{}".format(phase))

    @staticmethod
    def bordered_image(img: np.array, is_correct: bool):
        color = [0, 255, 0] if is_correct else [255, 0, 0]
        img[:20, :], img[-20:, :] = color, color
        img[:, :20], img[:, -20:] = color, color
        return Image.fromarray(img)
