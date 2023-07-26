"""
This script was created to evaluate the current traffickcam production model
against the current ViT model.
The current model is a ResNet-50. The output of this script will return the
accuracies for validation at k nearest neighbors as well as the duplicates
found. The models are evaluated on Hotels-50K dataset.
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data

from timm import models
import faiss
import glob
import math
import random

from torch.backends import cudnn

from traffickcam_folder_50k import TraffickcamFolderPaths_50k
from train import AccCalculator

checkpoint_path = '/shared/data/Traffickcam/resnet50-hardnegative-02152021.pth'
checkpoint_path_vit = '/home/tun78940/tcam/tcam_training/traffickcam_model_training/models/latest_25_648000_checkpoint.pth.tar'


def main():
    device = torch.device("cuda")
    print(device)
    print("HOTELS-50K")
    # Eval ResNet-50 model for accuracy metrics on Hotels-50K
    # print("Evaluating accuracy for ResNet-50")
    # get_resnet()

    # Eval ViT model for accuracy metrics on Hotels-50K
    print("\n Evaluating accuracy for ViT")
    print("Evaluating from checkpoint:", checkpoint_path_vit)
    get_vit()


def get_resnet():
    model = Model(is_bulk=True)

    train_set = glob.glob("/shared/data/Hotels-50K/images/train/*/*/*/*")
    query_set = glob.glob('/shared/data/Hotels-50K/images/test/*/*/*/*')

    # train_set = random.sample(train_set, 1000)
    # query_set = random.sample(query_set, 1000)

    # Load the dataset through TraffickCamFolderPaths to retrieve paths, labels and targets
    train_folder = TraffickcamFolderPaths_50k(train_set, transform=transforms.Compose(model.transform),
                                          camera_type_dict=None)
    query_folder = TraffickcamFolderPaths_50k(query_set, classes=train_folder.classes,
                                              transform=transforms.Compose(model.transform))

    # Setup DataLoaders for torch to iterate through
    train_loader = torch.utils.data.DataLoader(train_folder, batch_size=model.batch_sz, shuffle=False,
                                                   num_workers=model.num_workers, pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_folder, batch_size=model.batch_sz, shuffle=False,
                                                 num_workers=model.num_workers, pin_memory=True)

    loaders_dict = {'train': train_loader, 'query': query_loader}

    accuracies_dict = {'val': {}}

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

    eval_model(model.model, loaders_dict, accuracies_dict, acc_calculator)


def get_vit():
    # Setup model and declare GPU & checkpoint usage
    model = models.vit_base_patch16_224_in21k(pretrained=True)
    model.head = torch.nn.Identity()
    model.cuda()

    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path_vit)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    cudnn.benchmark = True

    # Define transformations
    test_transform = [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ]
    # Specify image transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    rotate = transforms.RandomApply([transforms.RandomRotation((-35, 35))], p=.2)
    color_jitter = transforms.RandomApply([transforms.ColorJitter(brightness=.25, hue=.15, saturation=.05)], p=.4)
    train_transforms = [transforms.Resize(224), transforms.RandomCrop(224),
                        rotate, color_jitter, transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), normalize]

    # Get our data from Hotels-50K
    train_set = glob.glob("/shared/data/Hotels-50K/images/train/*/*/*/*")
    query_set = glob.glob("/shared/data/Hotels-50K/images/test/*/*/*/*")

    # train_set = random.sample(train_set, 1000)
    # query_set = random.sample(query_set, 1000)

    train_folder = TraffickcamFolderPaths_50k(train_set, transform=transforms.Compose(train_transforms),
                                          camera_type_dict=None)
    query_folder = TraffickcamFolderPaths_50k(query_set, classes=train_folder.classes,
                                          transform=transforms.Compose(test_transform))

    # Setup DataLoaders for torch to iterate through
    train_loader = torch.utils.data.DataLoader(train_folder, batch_size=128, shuffle=False,
                                             num_workers=4, pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_folder, batch_size=128, shuffle=False,
                                               num_workers=4, pin_memory=True)

    loaders_dict = {'query': query_loader, 'train': train_loader}

    accuracies_dict = {'val': {}}

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

    eval_model(model, loaders_dict, accuracies_dict, acc_calculator)


def eval_model(model, loaders, accuracy_dict, acc_calculator):
    print("evaluating model")

    # Get gallery_embeddings to compare with validation set
    gal_embeddings, gal_labels, gal_paths = embed(loaders['train'], model)

    print("Got Gal embeddings")

    query_embeddings, query_labels, query_paths = embed(loaders['query'], model)
    print("Got query embeddings")
    accuracies, knn_labels = get_accuracies(acc_calculator, gal_embeddings, query_embeddings, gal_labels, query_labels)

    accuracy_dict['val'][0] = accuracies
    log_accuracies('val', accuracies[:3])
    print("Validation Duplicates:", accuracies[3])
    print('Val accuracy: {}'.format(accuracies))

    torch.cuda.empty_cache()


# def embed(data_loader, model):
#     all_embeddings = torch.Tensor([])
#     all_labels = torch.Tensor([])
#     all_paths = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, (inputs, labels, paths) in enumerate(data_loader):
#             inputs = inputs.cuda()
#             embeddings = model(inputs)
#             all_embeddings = torch.cat((all_embeddings, embeddings.detach().cpu()))
#             all_labels = torch.cat((all_labels, labels))
#             all_paths = all_paths + paths
#     return all_embeddings.numpy(), all_labels.numpy(), all_paths

# Optimized embedding function
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

def log_accuracies(phase, accuracies):
    print("{}/acc =".format(phase), accuracies[0])
    for idx, accuracy in enumerate(accuracies):
        k = [1, 10, 100]
        print("{}/retrieval_at_{} = ".format(phase, k[idx]), accuracy)


def get_accuracies(acc_calculator, ref_embeddings, query_embeddings, ref_labels, query_labels,
                   embeddings_come_from_same_source=False):
    faiss.normalize_L2(ref_embeddings)
    faiss.normalize_L2(query_embeddings)

    batch_size = 100
    num_chunks = int(math.ceil(query_embeddings.shape[0] / batch_size))

    retrieval_results = []
    knn_labels_list = []

    for i in range(num_chunks):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        query_embeddings_chunk = query_embeddings[start_idx:end_idx]
        query_labels_chunk = query_labels[start_idx:end_idx]

        # Calculate accuracies for the current chunk
        accuracies = acc_calculator.get_accuracy(query_embeddings_chunk,
                                                 query_labels_chunk,
                                                 ref_embeddings,
                                                 ref_labels,
                                                 embeddings_come_from_same_source)

        retrieval_results.append(torch.tensor([accuracies['retrieval_at_1'],
                                  accuracies['retrieval_at_10'],
                                  accuracies['retrieval_at_100'],
                                  accuracies['duplicates']]))
        knn_labels_list.append(accuracies['knn_labels'])

    # Concatenate the results from all the chunks
    retrieval_results = torch.stack(retrieval_results)
    knn_labels = torch.cat(knn_labels_list)

    # Calculate the overall accuracy for each retrieval since we did it in chunks
    overall_accuracy = retrieval_results.mean(dim=0)

    return overall_accuracy, knn_labels


class Model:
    def __init__(self, is_bulk):
        # Initialize the network graph and loads the weights
        # if is_bulk is set to True, use the GPU implementation
        self.name = 'ResNet'
        self.output_sz = 256
        self.image_sz = 256
        self.batch_sz = 1
        self.num_workers = 0
        self.device = 'cpu'
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, self.output_sz)
        self.model.load_state_dict(torch.load(checkpoint_path))
        if is_bulk:
            self.batch_sz = 100
            self.num_workers = 4
            self.device = 'cuda'
            self.model = torch.nn.DataParallel(self.model).cuda()
        normalize = transforms.Normalize(mean=[0.5838, 0.5146, 0.4470], std=[0.6298, 0.6112, 0.4445])
        self.transform = [transforms.Resize(self.image_sz), transforms.CenterCrop(self.image_sz-32), transforms.ToTensor(), normalize]

    def eval(self):
        # Define eval for pytorch usage
        self.model.eval()


if __name__ == "__main__":
    main()
