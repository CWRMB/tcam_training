"""
This script is used for evaluating and embedding models from checkpoints.
This script will return the image paths of nearest neighbors to a specified query image
By default the query image will be randomly selected. This is meant to be used along with a
Jupyter Notebook to then plot the images and view them.
"""
import glob

import torch
from torchvision import transforms
import torch.utils.data

import numpy as np
import pickle
import faiss
from timm import models
import random

from collections import OrderedDict
from traffickcam_folder_50k import TraffickcamFolderPaths_50k
from traffickcam_folder_50k import _extract_ids
from resnet_eval import Model

checkpoint_path_vit = "/home/tun78940/tcam/tcam_training/traffickcam_model_training/models/latest_25_648000_checkpoint.pth.tar"
checkpoint_path = '/shared/data/Traffickcam/resnet50-hardnegative-02152021.pth'

# Hard coded after we get the query image from running vit first
query_set_cnn = ["/shared/data/Hotels-50K/images/val_query/91/124897/travel_website/8992072.jpg"]

def main():
    device = torch.device("cuda")
    print(device)

    eval_vit()
    #eval_cnn()


def get_distance(query_embed, val_embed, val_paths):
    # Get dimension of vector and build index
    d = query_embed.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(val_embed)

    # Nearest neighbor search
    k = 5
    D, I = index.search(query_embed, k)

    # Get the image path of the nearest neighbor
    nearest_paths = [val_paths[idx] for idx in I[0]]

    return D, I, nearest_paths


def embed(data_loader, model):
    # Embed function to get the paths and labels
    all_embeddings = torch.Tensor([])
    all_labels = torch.Tensor([])
    all_paths = []

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(data_loader):
            inputs = inputs.cuda()
            embeddings = model(inputs)
            all_embeddings = torch.cat((all_embeddings, embeddings.detach().cpu()))
            all_labels = torch.cat((all_labels, labels))
            all_paths = all_paths + paths
    return all_embeddings.numpy(), all_labels.numpy(), all_paths


def eval_cnn():
    model = Model(is_bulk=True)

    train_set = glob.glob("/shared/data/Hotels-50K/images/train/*/*/travel_website/*")
    val_set = glob.glob("/shared/data/Hotels-50K/images/val/*/*/travel_website/*")

    # Hard coded after we get the query image from running vit first
    query_set = query_set_cnn

    # TODO clean up duplicated code (did this so its easier to use both models)

    train_folder = TraffickcamFolderPaths_50k(train_set, transform=transforms.Compose(model.transform),
                                          camera_type_dict=None)
    val_folder = TraffickcamFolderPaths_50k(val_set, classes=train_folder.classes,
                                        transform=transforms.Compose(model.transform))
    query_folder = TraffickcamFolderPaths_50k(query_set, classes=train_folder.classes,
                                          transform=transforms.Compose(model.transform))

    # Setup DataLoaders for torch to iterate through
    query_loader = torch.utils.data.DataLoader(query_folder, batch_size=model.batch_sz, shuffle=False,
                                               num_workers=model.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_folder, batch_size=model.batch_sz, shuffle=False,
                                             num_workers=model.num_workers, pin_memory=True)

    loaders_dict = {'query': query_loader, 'val': val_loader}

    # Get embeddings
    val_embeddings, test_labels, test_paths = embed(loaders_dict['val'], model.model)
    query_embeddings, val_labels, val_paths = embed(loaders_dict['query'], model.model)

    distances, indices, nearest_paths = get_distance(query_embeddings, val_embeddings, val_set)

    print("Distances:", distances)
    print("Indices:", indices)

    img_id, hotel_id = _extract_ids(query_set[0])
    print("Query path:", query_set[0], " ", f"IMG ID:{img_id}", f"Hotel ID:{hotel_id}")

    print("Nearest neighbor paths")
    for path in nearest_paths:
        img_id, hotel_id = _extract_ids(path)
        print(path, " ", f"IMG ID:{img_id}", f"Hotel ID:{hotel_id}")


def eval_vit():
    # Setup model and declare GPU & checkpoint usage
    model = models.vit_base_patch16_224_in21k(pretrained=True)
    model.head = torch.nn.Identity()
    model.cuda()

    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path_vit)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    # Specify image transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    rotate = transforms.RandomApply([transforms.RandomRotation((-35, 35))], p=.2)
    color_jitter = transforms.RandomApply([transforms.ColorJitter(brightness=.25, hue=.15, saturation=.05)], p=.4)
    train_transforms = [transforms.Resize(224), transforms.RandomCrop(224),
                        rotate, color_jitter, transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), normalize]
    test_transform = [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                      normalize]

    train_set = glob.glob("/shared/data/Hotels-50K/images/train/*/*/travel_website/*")
    val_set = glob.glob("/shared/data/Hotels-50K/images/val/*/*/travel_website/*")
    query_set = glob.glob("/shared/data/Hotels-50K/images/val_query/*/*/travel_website/*")

    # Get one query image to embed
    query_set = random.sample(query_set, 1)

    train_folder = TraffickcamFolderPaths_50k(train_set, transform=transforms.Compose(train_transforms),
                                          camera_type_dict=None)
    val_folder = TraffickcamFolderPaths_50k(val_set, classes=train_folder.classes,
                                         transform=transforms.Compose(test_transform))
    query_folder = TraffickcamFolderPaths_50k(query_set, classes=train_folder.classes,
                                        transform=transforms.Compose(test_transform))

    # Setup DataLoaders for torch to iterate through
    query_loader = torch.utils.data.DataLoader(query_folder, batch_size=256, shuffle=False,
                                                   num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_folder, batch_size=256, shuffle=False,
                                                 num_workers=4, pin_memory=True)

    loaders_dict = {'query': query_loader, 'val': val_loader}

    # Get embeddings
    val_embeddings, test_labels, test_paths = embed(loaders_dict['val'], model)
    query_embeddings, val_labels, val_paths = embed(loaders_dict['query'], model)

    distances, indices, nearest_paths = get_distance(query_embeddings, val_embeddings, val_set)

    # print("Distances:", distances)
    # print("Indices:", indices)

    img_id, hotel_id = _extract_ids(query_set[0])
    print("Query path:", query_set[0], " ", f"IMG ID:{img_id}", f"Hotel ID:{hotel_id}")

    print("Nearest neighbor paths")
    for path in nearest_paths:
        img_id, hotel_id = _extract_ids(path)
        print(path, " ", f"IMG ID:{img_id}", f"Hotel ID:{hotel_id}")


if __name__ == "__main__":
    main()
