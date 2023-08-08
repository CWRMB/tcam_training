"""
This class file will gather images from the Traffickcam mini dataset.
This dataset is located at (/shared/data/tcam_mini). This dataset
contains 100 images to test the model on.
"""

import glob



class LoadTcamMini:
    def __init__(self, directory, total_images=100, train_ratio=0.6, train_query_ratio=0.2, gallery_ratio=0.1, validation_ratio=0.1):
        self.directory = directory
        self.total_images = total_images
        self.train_ratio = train_ratio
        self.train_query_ratio = train_query_ratio
        self.gallery_ratio = gallery_ratio
        self.validation_ratio = validation_ratio

    def set_splits(self):
        num_train = int(self.total_images * self.train_ratio)
        num_train_query = int(self.total_images * self.train_query_ratio)
        num_gallery = int(self.total_images * self.gallery_ratio)
        num_validation = int(self.total_images * self.validation_ratio)

        self.directory = self.directory + "/*"

        # Expects all the images to be .jpg format and be located within a single directory folder
        dataset = glob.glob(self.directory)
        
        # Could do a random shuffle of the dataset but for consistency I am keeping it the same
        train_set = dataset[:num_train]
        train_query_set = dataset[num_train:num_train+num_train_query]
        gallery_set = dataset[num_train+num_train_query:num_train+num_train_query+num_gallery]
        validation_set = dataset[num_train+num_train_query+num_gallery:]

        return train_set, train_query_set, gallery_set, validation_set

