"""
This script will calculate accuracy metrics of the
ViT model at saved checkpoints. This is done sice we
are now separating the evaluation step from the
training.
"""

import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data
import torch.backends.cudnn as cudnn

from timm import *
import faiss
import time
import numpy as np

import pickle
import os
import neptune

from traffickcam_folder import TraffickcamFolderPaths
from pytorch_metric_learning.utils import accuracy_calculator
from resnet_eval import Model


checkpoint_path = '/home/tun78940/tcam/tcam_training/traffickcam_model_training/models/latest_25_648000_checkpoint.pth.tar'

training_images = '/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/train_imgs.dat'
gallery_imgs = '/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/gallery_imgs.dat'
val_query_images = '/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/validation_queries.dat'
train_query_images = '/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/train_queries.dat'

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in [0, 1])
    device = torch.device("cuda")
    print(device)

    start = time.time()
    #get_vit()
    get_resnet()

    end = time.time()
    print("Elapsed time: {}".format(end - start))


def get_vit():
    # os.environ[
    #     'NEPTUNE_API_TOKEN'] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzhhZjM5OS1kZjdjLTQ3MzAtODcyMS0yN2JiMWQyNDhhMGYifQ=="

    # logger = neptune.init_run(project=args.name, with_id="TCAM-58")

    print("Evaluating from checkpoint:", checkpoint_path)
    print("Evaluating ViT Triplet Margin")

    model = models.vit_base_patch16_224_in21k(pretrained=True)
    model.head = torch.nn.Identity()
    model.cuda()

    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    cudnn.benchmark = True

    # Specify image transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    rotate = transforms.RandomApply([transforms.RandomRotation((-35, 35))], p=.2)
    color_jitter = transforms.RandomApply([transforms.ColorJitter(brightness=.25, hue=.15, saturation=.05)], p=.4)
    train_transforms = [transforms.Resize(224), transforms.RandomCrop(224),
                        rotate, color_jitter, transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), normalize]
    test_transforms = [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]


    # load pickle lists containing the paths to each image set
    with open(training_images, 'rb') as f:
        train_set = pickle.load(f)
        f.close()
    with open(gallery_imgs, 'rb') as f:
        gallery_images = pickle.load(f)
        f.close()
    with open(val_query_images, 'rb') as f:
        val_queries = pickle.load(f)
        f.close()
    with open(train_query_images, 'rb') as f:
        train_queries = pickle.load(f)
        f.close()

    # Skip .txt invalid image formats within the file directory
    train_set = [file for file in train_set if not file.endswith(".txt")]
    gallery_images = [file for file in gallery_images if not file.endswith(".txt")]
    val_queries = [file for file in val_queries if not file.endswith(".txt")]
    train_queries = [file for file in train_queries if not file.endswith(".txt")]

    train_folder = TraffickcamFolderPaths(train_set, transform=transforms.Compose(train_transforms),
                                          camera_type_dict=None)
    val_query_folder = TraffickcamFolderPaths(val_queries, classes=train_folder.classes,
                                              transform=transforms.Compose(test_transforms))
    train_query_folder = TraffickcamFolderPaths(train_queries, classes=train_folder.classes,
                                                transform=transforms.Compose(test_transforms))
    gallery_folder = TraffickcamFolderPaths(gallery_images, classes=train_folder.classes,
                                            transform=transforms.Compose(test_transforms))

    print('Train folders created')

    val_query_loader = torch.utils.data.DataLoader(val_query_folder, batch_size=256, shuffle=False,
                                               num_workers=10, pin_memory=True)
    train_query_loader = torch.utils.data.DataLoader(train_query_folder, batch_size=256, shuffle=False,
                                                 num_workers=10, pin_memory=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_folder, batch_size=256, shuffle=False,
                                             num_workers=10, pin_memory=True)

    loaders_dict = {'val_query': val_query_loader, 'train_query': train_query_loader,
                    'gallery': gallery_loader}

    acc_calculator = AccCalculator(include=(
        "precision_at_1",
        "precision_at_5",
        "precision_at_10",
        "retrieval_at_1",
        "retrieval_at_10",
        "retrieval_at_100",
        "duplicates",
        "knn_labels"),
        k=100)
    #eval_model(loaders_dict, model, acc_calculator, logger)
    eval_model(loaders_dict, model, acc_calculator)


def get_resnet():
    model = Model(is_bulk=True)

    print("Evaluating CNN ResNet Model on new embed function")

    # load pickle lists containing the paths to each image set
    with open(training_images, 'rb') as f:
        train_set = pickle.load(f)
        f.close()
    with open(gallery_imgs, 'rb') as f:
        gallery_images = pickle.load(f)
        f.close()
    with open(val_query_images, 'rb') as f:
        val_queries = pickle.load(f)
        f.close()

    cudnn.benchmark = True

    # Load the dataset through TraffickCamFolderPaths to retrieve paths, labels and targets
    train_folder = TraffickcamFolderPaths(train_set, transform=transforms.Compose(model.transform),
                                          camera_type_dict=None)
    gallery_folder = TraffickcamFolderPaths(gallery_images, classes=train_folder.classes,
                                            transform=transforms.Compose(model.transform))
    query_folder = TraffickcamFolderPaths(val_queries, classes=train_folder.classes,
                                              transform=transforms.Compose(model.transform))

    # Setup DataLoaders for torch to iterate through
    train_loader = torch.utils.data.DataLoader(train_folder, batch_size=model.batch_sz, shuffle=False,
                                                   num_workers=model.num_workers, pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_folder, batch_size=model.batch_sz, shuffle=False,
                                                 num_workers=model.num_workers, pin_memory=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_folder, batch_size=model.batch_sz, shuffle=False,
                                                 num_workers=model.num_workers, pin_memory=True)

    loaders_dict = {'train': train_loader, 'val_query': query_loader, 'gallery': gallery_loader}


    acc_calculator = AccCalculator(include=(
                                        "precision_at_1",
                                        "precision_at_5",
                                        "precision_at_10",
                                        "retrieval_at_1",
                                        "retrieval_at_10",
                                        "retrieval_at_100",
                                        "duplicates",
                                        "knn_labels"),
                                    k=100)

    # Evaluation Step
    gal_embeddings, gal_labels, gal_paths = embed(loaders_dict['gallery'], model.model)
    query_embeddings, query_labels, query_paths = embed(loaders_dict['val_query'], model.model)
    accuracies, knn_labels = get_accuracies(acc_calculator, gal_embeddings, query_embeddings, gal_labels,
                                            query_labels)

    print('Val accuracy: {}'.format(accuracies))


# TODO PUT PARAM LOGGER BACK and fix logging when ready
def eval_model(loaders, model, acc_calculator):
    print("Evaluating model")

    query_embeddings, query_labels, query_paths = embed(loaders['train_query'], model)
    gal_embeddings, gal_labels, gal_paths = embed(loaders['gallery'], model)
    accuracies, knn_labels = get_accuracies(acc_calculator, gal_embeddings, query_embeddings, gal_labels,
                                            query_labels)

    #log_accuracies('train', accuracies[:3], logger)
    #logger["Train/duplicates"].append(accuracies[3])
    print_accuracies("train", accuracies[:3])
    torch.cuda.empty_cache()

    query_embeddings, query_labels, query_paths = embed(loaders['val_query'], model)
    accuracies, knn_labels = get_accuracies(acc_calculator, gal_embeddings, query_embeddings, gal_labels,
                                            query_labels)

    #log_accuracies('val', accuracies[:3], logger)
    #logger["Val/duplicates"].append(accuracies[3])
    print_accuracies("val", accuracies[:3])

    torch.cuda.empty_cache()


def embed(data_loader, model):
    num_images = len(data_loader.dataset)
    if model.module.__class__.__name__ == 'ResNet':
        print("Got ResNet")
        embed_size = 256
    else:
        print("Got ViT")
        embed_size = model.module.embed_dim

    all_embeddings = torch.zeros(num_images, embed_size)
    all_labels = torch.zeros(num_images)
    all_paths = []

    model.eval()
    with torch.no_grad():
        start_index = 0
        for i, (inputs, labels, paths) in enumerate(data_loader):
            inputs = inputs.cuda()
            batch_size = inputs.size(0)

            embeddings = model(inputs)

            all_embeddings[start_index : start_index + batch_size] = embeddings.detach().cpu()
            all_labels[start_index : start_index + batch_size] = labels
            all_paths.extend(paths)

            start_index += batch_size

    return all_embeddings.numpy(), all_labels.numpy(), all_paths


def get_accuracies(acc_calculator, ref_embeddings, query_embeddings, ref_labels, query_labels,
                   embeddings_come_from_same_source=False):
    faiss.normalize_L2(ref_embeddings)
    faiss.normalize_L2(query_embeddings)
    accuracies = acc_calculator.get_accuracy(query_embeddings,
                                             query_labels,
                                             ref_embeddings,
                                             ref_labels,
                                             embeddings_come_from_same_source)
    # just return retrieval for now
    return [accuracies['retrieval_at_1'],
            accuracies['retrieval_at_10'],
            accuracies['retrieval_at_100'],
            accuracies['duplicates']], accuracies['knn_labels']


def log_accuracies(phase, accuracies, logger):
    logger["{}/acc".format(phase)].append(accuracies[0])
    for idx, accuracy in enumerate(accuracies):
        k = [1, 10, 100]
        logger["{}/retrieval_at_{}".format(phase, k[idx])].append(accuracy)


def print_accuracies(phase, accuracies):
    print("{}/acc:".format(phase), accuracies[0])
    for idx, accuracy in enumerate(accuracies):
        k = [1, 10, 100]
        print("{}/retrieval_at_{}".format(phase, k[idx]), accuracy)


class AccCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_1(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 1, self.avg_of_avgs,
                                                  return_per_class=False, label_comparison_fn=torch.eq)

    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 5, self.avg_of_avgs,
                                                  return_per_class=False, label_comparison_fn=torch.eq)

    def calculate_precision_at_10(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 10, self.avg_of_avgs,
                                                  return_per_class=False, label_comparison_fn=torch.eq)

    def calculate_knn_labels(self, knn_labels, query_labels, **kwargs):
        return knn_labels

    def retrieval_at_k(self, k, knn_labels, query_labels):
        curr_knn_labels = knn_labels[:, :k]
        accuracy_per_sample = np.apply_along_axis(any, axis=1, arr=(curr_knn_labels == query_labels[:, None]).cpu())
        accuracy_per_sample = torch.tensor(accuracy_per_sample).to(query_labels.device).float()
        return accuracy_calculator.maybe_get_avg_of_avgs(accuracy_per_sample, query_labels, self.avg_of_avgs,
                                                         return_per_class=False)

    def calculate_retrieval_at_1(self, knn_labels, query_labels, **kwargs):
        return self.retrieval_at_k(1, knn_labels, query_labels)

    def calculate_retrieval_at_10(self, knn_labels, query_labels, **kwargs):
        return self.retrieval_at_k(10, knn_labels, query_labels)

    def calculate_retrieval_at_100(self, knn_labels, query_labels, **kwargs):
        return self.retrieval_at_k(100, knn_labels, query_labels)

    def calculate_duplicates(self, knn_distances, **kwargs):
        duplicates = knn_distances[:, :1].squeeze(1) == 0
        return sum(duplicates) / len(duplicates)

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_1", "precision_at_5", "precision_at_10", "retrieval_at_1",
                                         "retrieval_at_10", "retrieval_at_100", "knn_labels"]


if __name__ == "__main__":
    main()
