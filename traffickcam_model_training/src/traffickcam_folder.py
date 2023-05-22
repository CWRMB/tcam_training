import os

import torch
from torchvision.datasets.folder import default_loader


def _extract_ids(im_path):
    img_id, hotel_id = im_path.split(os.sep)[-1].split('.')[0].split('_')
    return img_id, hotel_id


class TraffickcamFolderPaths(torch.utils.data.Dataset):

    def __init__(self, paths, classes=None, transform=None, camera_type_dict=None):

        self.paths = paths

        if classes is None:
            self.classes, self.class_to_idx = self._find_classes()
        else:
            self.classes = classes
            self.class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = self._make_dataset()
        self.targets = [s[1] for s in self.samples]

        self.camera_type_dict = camera_type_dict
        if self.camera_type_dict:
            self.capture_method_ids = [self.camera_type_dict[_extract_ids(s[0])[0]] for s in self.samples]
        else:
            self.capture_method_ids = None

        self.transform = transform

    def _find_classes(self):
        classes = set()
        for path in self.paths:
            _, hotel_id = _extract_ids(path)
            classes.add(hotel_id)
        classes = list(classes)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        num_missing = 0
        for path in self.paths:
            _, hotel_id = _extract_ids(path)
            if hotel_id in self.class_to_idx:
                item = (path, self.class_to_idx[hotel_id])
                samples.append(item)
            else:
                num_missing += 1
        print(num_missing, "hotels missing out of", len(samples))
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path

    def __len__(self):
        return len(self.samples)