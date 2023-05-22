import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace)


def generate_dictionary(labels, capture_method_ids):
    d = {}
    for i in range(len(labels)):
        if labels[i] not in d:
            d[labels[i]] = {'tcam': [], 'exp': []}
        if capture_method_ids[i] == 1:
            capture_method = 'tcam'
        else:
            capture_method = 'exp'
        d[labels[i]][capture_method].append(i)
    return d


class TraffickcamSampler(Sampler):

    def __init__(self, labels, capture_method_ids, m, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if isinstance(capture_method_ids, torch.Tensor):
            capture_method_ids = capture_method_ids.numpy()
        assert len(labels) == len(capture_method_ids), 'These arrays must be the same length'

        self.m_per_class = int(m)
        self.labels_to_capture_method_to_indices = generate_dictionary(labels, capture_method_ids)
        self.labels = list(self.labels_to_capture_method_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.length_of_single_pass < self.list_size:
            self.list_size -= (self.list_size) % (self.length_of_single_pass)

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.list_size // self.length_of_single_pass if self.length_of_single_pass < self.list_size else 1
        for _ in range(num_iters):
            np.random.shuffle(self.labels)
            for label in self.labels:
                exp_images, tcam_images = self.labels_to_capture_method_to_indices[label]['exp'],\
                                          self.labels_to_capture_method_to_indices[label]['tcam']

                if exp_images:
                    if tcam_images:
                        num_tcam = self.m_per_class // 2
                        num_exp = self.m_per_class // 2
                    else:
                        num_tcam = 0
                        num_exp = self.m_per_class
                else:
                    num_tcam = self.m_per_class // 2
                    num_exp = 0

                idx_list[i: i + num_tcam] = safe_random_choice(tcam_images, size=num_tcam)
                i += num_tcam
                idx_list[i: i + num_exp] = safe_random_choice(exp_images, size=num_exp)
                i += num_exp
        return iter(idx_list)