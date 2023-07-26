import argparse
import os
import shutil
import time

import pickle
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from timm import *
import random

from pytorch_metric_learning import miners, losses, samplers
from pytorch_metric_learning.utils import accuracy_calculator
import faiss
import neptune

from traffickcam_folder import TraffickcamFolderPaths
from traffickcam_sampler import TraffickcamSampler
import loss
import visualizations


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch Traffickcam Training')
parser.add_argument('--training_images',
                    default='/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/train_imgs.dat', type=str,
                    help='pickle list of images used to train model')
parser.add_argument('--val_query_images',
                    default='/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/validation_queries.dat',
                    type=str,
                    help='pickle list of validation images used for queries to measure accuracy')
parser.add_argument('--train_query_images',
                    default='/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/train_queries.dat',
                    type=str,
                    help='set of training images used for queries to measure accuracy')
parser.add_argument('--gallery_images',
                    default='/home/tun78940/tcam/tcam_training/traffickcam_model_training/src/gallery_imgs.dat',
                    type=str,
                    help='set of training images that are used in the gallery to measure train and validation accuracy')
parser.add_argument('--capture_id_file', default=None, type=str,
                    help='Pandas DF for image capture')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=32, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--percent_masked', default=.4, type=float,
                    help='percent of each image masked during training')
parser.add_argument('--batch_duplication', default=False, type=bool,
                    help='whether or not to duplicate batches with masked images')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (before duplication)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--compute_accuracy_freq', default=6000, type=int,
                    help='after this many global steps accuracy is computed and model saved')
parser.add_argument('--resume',
                    default='/home/tun78940/tcam/tcam_training/traffickcam_model_training/models/latest_checkpoint.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', const=True, default=True, type=str2bool, nargs='?',
                    help='use pre-trained model')
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='Whether or not to use GPUs')
parser.add_argument('--gpu', type=int, default=[0, 1], nargs='+',
                    help='GPUs to use')
parser.add_argument('--m', default=10, type=int,
                    help='num class samples per batch')
parser.add_argument('--margin', default=0.1, type=float,
                    help='triplet loss margin')
parser.add_argument('--keep_opt_parameters', const=True, default=False, type=str2bool, nargs='?',
                    help='use optimizer parameters loaded in checkpoint (if applicable) or use those specified in arguments')
parser.add_argument('--name', default='vidarlab/tcamTraining',
                    help='Experiment name for logger')
parser.add_argument('--tags', default=['FullTraffickCam'], type=str, nargs='*',
                    help='Tags to be sent to logger')
parser.add_argument('--loss', default='triplet_margin_loss', type=str,
                    help='loss to be used in training, choose one of {}'.format(loss.names()))
parser.add_argument('--model', default='vit_base_patch16', type=str,
                    help='model to be used in training, choose one of {}'
                    .format([model for model in list(models.__dict__) if model[0] != "_"]))
parser.add_argument('--tsne', default=False, type=bool,
                    help='whether or not to generate and log t-SNE plots after evaluation')
parser.add_argument('--knn_images', default=0, type=int,
                    help='the number of KNN images to log')
parser.add_argument('--confusion_matrix', default=False, type=bool,
                    help='whether or not to generate and log confusion matrices after evaluation')
parser.add_argument('--input_size', default=224, type=int,
                    help='size of input images')
parser.add_argument('--resize', default=256, type=int,
                    help='resize image in transforms')


def main():
    # torch.cuda.empty_cache()
    global args, global_step
    global_step = 1
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    device = torch.device("cuda" if args.use_gpu else "cpu")
    print(device)
    os.environ[
        'NEPTUNE_API_TOKEN'] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzhhZjM5OS1kZjdjLTQ3MzAtODcyMS0yN2JiMWQyNDhhMGYifQ=="

    model = models.vit_base_patch16_224_in21k(pretrained=True)

    model.head = torch.nn.Identity()

    model.cuda()

    model = torch.nn.DataParallel(model)
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print("GPUS", available_gpus)
    accuracies_dict = {'train': {}, 'val': {}}

    loss_func = loss.create(args.loss, margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # anything that can be modified with arguments
    params = {
        'learning_rate': args.lr,
        'pretrained': args.pretrained,
        'optimizer': 'Adam',
        'margin': args.margin,
        'batch_size': args.batch_size,
        'group_size': args.m,
        'percent_masked': args.percent_masked if args.batch_duplication else None,
        'loss': args.loss,
        'model': args.model,
        'using_batch_duplication': args.batch_duplication,
        'input_size': args.input_size,
    }

    # logger = neptune.init_run(project=args.name)
    logger = neptune.init_run(project=args.name, with_id="TCAM-58")
    logger["parameters"] = params

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            global_step = checkpoint['global_step']
            global_step += 1
            # accuracies_dict = checkpoint['accuracies']
            if args.keep_opt_parameters:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, global step {})"
                  .format(args.resume, checkpoint['epoch'], global_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Specify image transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    rotate = transforms.RandomApply([transforms.RandomRotation((-35, 35))], p=.2)
    color_jitter = transforms.RandomApply([transforms.ColorJitter(brightness=.25, hue=.15, saturation=.05)], p=.4)
    train_transforms = [transforms.Resize(args.resize), transforms.RandomCrop(args.input_size), rotate, color_jitter,
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), normalize]
    test_transforms = [transforms.Resize(args.resize), transforms.CenterCrop(args.input_size), transforms.ToTensor(), normalize]

    # load pickle lists containing the paths to each image set
    with open(args.training_images, 'rb') as f:
        train_set = pickle.load(f)
        f.close()
    with open(args.gallery_images, 'rb') as f:
        gallery_images = pickle.load(f)
        f.close()
    with open(args.val_query_images, 'rb') as f:
        val_queries = pickle.load(f)
        f.close()
    with open(args.train_query_images, 'rb') as f:
        train_queries = pickle.load(f)
        f.close()

    # load capture_id file and convert it to dictionary
    if args.capture_id_file:
        id_to_capture = {}
        with open(args.capture_id_file, 'rb') as f:
            df = pickle.load(f)
            f.close()
            ids, captures = df['id'].values, df['capture_method_id'].values
            for i in range(len(ids)):
                id_to_capture[str(ids[i])] = captures[i]
    else:
        id_to_capture = None

    # Skip .txt invalid image formats within the file directory
    train_set = [file for file in train_set if not file.endswith(".txt")]
    gallery_images = [file for file in gallery_images if not file.endswith(".txt")]
    val_queries = [file for file in val_queries if not file.endswith(".txt")]
    train_queries = [file for file in train_queries if not file.endswith(".txt")]

    train_folder = TraffickcamFolderPaths(train_set, transform=transforms.Compose(train_transforms),
                                          camera_type_dict=id_to_capture)
    val_query_folder = TraffickcamFolderPaths(val_queries, classes=train_folder.classes,
                                              transform=transforms.Compose(test_transforms))
    train_query_folder = TraffickcamFolderPaths(train_queries, classes=train_folder.classes,
                                                transform=transforms.Compose(test_transforms))
    gallery_folder = TraffickcamFolderPaths(gallery_images, classes=train_folder.classes,
                                            transform=transforms.Compose(test_transforms))

    print('Train folders created')
    if args.capture_id_file:
        # sample equal number of mobile-app and Expedia images per class per batch
        sampler = TraffickcamSampler(train_folder.targets, train_folder.capture_method_ids, args.m,
                                     length_before_new_iter=len(train_folder))
    else:
        # does not sample same number of mobile-app and Expedia images per class per batch
        sampler = samplers.MPerClassSampler(train_folder.targets, args.m, length_before_new_iter=len(train_folder))

    train_loader = torch.utils.data.DataLoader(train_folder, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                                               num_workers=args.workers, pin_memory=True)
    val_query_loader = torch.utils.data.DataLoader(val_query_folder, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True)
    train_query_loader = torch.utils.data.DataLoader(train_query_folder, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers, pin_memory=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_folder, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    print('Loaders created')
    loaders_dict = {'train': train_loader, 'val_query': val_query_loader, 'train_query': train_query_loader,
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

    visualizers = {
        'tsne': visualizations.create('tsne', logger) if args.tsne else None,
        'knn_images': visualizations.create('knn_images', logger, args.knn_images) if args.knn_images > 0 else None,
        'confusion_matrix': visualizations.create('confusion_matrix', logger) if args.confusion_matrix else None,
    }
    print('Beginning training')
    for epoch in range(args.start_epoch, args.epochs):
        train(model, epoch, loaders_dict, accuracies_dict, loss_func, optimizer, visualizers, acc_calculator, logger,
              device)


def generate_random_masks(data, percentage):
    h, w = data.shape[2], data.shape[3]
    mask = np.ones(shape=(h, w))
    local_mask_area = .05 * h * w
    sqr_root_local_mask_area = local_mask_area ** (1 / 2)
    while np.mean(mask) > 1 - percentage:
        local_mask_width = np.random.randint(low=int(sqr_root_local_mask_area / 2),
                                             high=int(sqr_root_local_mask_area * (3 / 2)))
        local_mask_height = int(local_mask_area / local_mask_width)
        top_left_pixel_y, top_left_pixel_x = np.random.randint(0, h - local_mask_height), np.random.randint(0,
                                                                                                            w - local_mask_width)
        mask[top_left_pixel_y: top_left_pixel_y + local_mask_height,
        top_left_pixel_x: top_left_pixel_x + local_mask_width] = 0
    return data * torch.tensor(mask, dtype=torch.float)


def train(model, epoch, loaders, accuracy_dict, loss_func, optimizer, visualizers, acc_calculator, logger, device):
    global global_step
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    print('Meters created')
    # switch to train mode
    model.train()
    print('Model switched to train mode')
    end = time.time()
    print('Loader enumerated')
    i = 0
    for input, target, _ in loaders['train']:
        # compute accuracy
        if global_step % args.compute_accuracy_freq == 0:
            # outputs = {
            #     'train': {},
            #     'validation': {},
            #     'gallery': {}
            # }
            #
            # print('Computing accuracy for global step: {}'.format(global_step))
            # is_best = False
            # query_embeddings, query_labels, query_paths = embed(loaders['train_query'], model)
            # gal_embeddings, gal_labels, gal_paths = embed(loaders['gallery'], model)
            # accuracies, knn_labels = get_accuracies(acc_calculator, gal_embeddings, query_embeddings, gal_labels,
            #                                         query_labels)
            #
            # print(gal_labels[0]), print(gal_paths[0])
            #
            # outputs['training'] = {
            #     'embeddings': query_embeddings,
            #     'labels': query_labels,
            #     'paths': query_paths,
            #     'knn_labels': knn_labels
            # }
            # outputs['gallery'] = {
            #     'embeddings': gal_embeddings,
            #     'labels': gal_labels,
            #     'paths': gal_paths
            # }
            #
            # accuracy_dict['train'][global_step] = accuracies
            # log_accuracies('train', accuracies[:3], logger)
            # logger["Train/duplicates"].append(accuracies[3])
            # print('Train accuracy: {}'.format(accuracies))
            # torch.cuda.empty_cache()
            #
            # k_index = 0  # Which accuracy@k to use to determine best model, 0 = R@1, 1 = R@10, 2 = R@100
            # best_acc = 0
            # prev_val_accuracies = accuracy_dict['val']
            # for entry in prev_val_accuracies:
            #     if prev_val_accuracies[entry][k_index] > best_acc:
            #         best_acc = prev_val_accuracies[entry][k_index]
            #
            # query_embeddings, query_labels, query_paths = embed(loaders['val_query'], model)
            # accuracies, knn_labels = get_accuracies(acc_calculator, gal_embeddings, query_embeddings, gal_labels,
            #                                         query_labels)
            # outputs['validation'] = {
            #     'embeddings': query_embeddings,
            #     'labels': query_labels,
            #     'paths': query_paths,
            #     'knn_labels': knn_labels
            # }
            #
            # accuracy_dict['val'][global_step] = accuracies
            # log_accuracies('val', accuracies[:3], logger)
            # logger["Val/duplicates"].append(accuracies[3])
            # print('Val accuracy: {}'.format(accuracies))
            # if accuracies[k_index] > best_acc:
            #     is_best = True
            torch.cuda.empty_cache()
            #
            print('Saving model for global step {}'.format(global_step))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step
            }, logger, is_best=False)
            #
            # for visualizer in visualizers.values():
            #     if visualizer is not None:
            #         try:
            #             visualizer.log(outputs)
            #         except Exception as e:
            #             print("Something went wrong with", visualizer)
            #             print(e)
            #
            # model.train()

        data_time.update(time.time() - end)

        # batch duplication with occluded masks
        if args.batch_duplication:
            masked_batch = generate_random_masks(input, args.percent_masked)
            input = torch.cat([input, masked_batch])
            target = target.repeat(2)
        target = target.to(device)
        input = input.to(device)
        # compute output
        model_output = model(input)
        loss = loss_func(model_output, target)

        # record loss
        logger["Train/loss"].append(loss.item())
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        global_step += 1

        torch.cuda.empty_cache()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loaders['train']), batch_time=batch_time,
                data_time=data_time, loss=losses)
            )
        i += 1

    print('Epoch completed')
    return losses.avg


def embed(data_loader, model):
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


def save_checkpoint(state, logger, is_best, filename='checkpoint.pth.tar'):
    #torch.save(state, './models/latest_{}'.format(filename))
    torch.save(state, './models/latest_{}_{}_{}'.format(state['epoch'], state['global_step'], filename))
    if is_best:
        shutil.copyfile('./models/latest_{}'.format(filename), './models/best_{}'.format(filename))



def log_accuracies(phase, accuracies, logger):
    logger["{}/acc".format(phase)].append(accuracies[0])
    for idx, accuracy in enumerate(accuracies):
        k = [1, 10, 100]
        logger["{}/retrieval_at_{}".format(phase, k[idx])].append(accuracy)


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SqueezeLastLayer(torch.nn.Module):
    def __init__(self):
        super(SqueezeLastLayer, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


if __name__ == '__main__':
    main()