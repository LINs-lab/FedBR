# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from numpy import int16
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder, CIFAR10, CIFAR100
from torchvision.transforms.functional import rotate
import torch.distributions.dirichlet as dirichlet
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
import random
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    "RotatedCIFAR10",
    "RotatedCIFAR100",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

def divide_by_label(dataset, class_num = 10):
    index_map = [[] for i in range(class_num)]
    len_map = [0 for _ in range(class_num)]
    for i in range(len(dataset)):
        index_map[dataset[i][1]].append(i)
        len_map[dataset[i][1]] += 1
    return index_map, len_map

def reweight(q, empty_class):
    # sum_q = sum(q)
    q[empty_class] = 0
    q = q / sum(q)
    return q


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']



class MultipleEnvironmentCifar100(MultipleDomainDataset):

    def get_noniid_class_and_labels(self, original_images, original_labels, N):
    
        M = len(original_labels) // N
        K = 1
        M = M * K

        clients_images = [[] for _ in range(N)]
        clients_labels = [[] for _ in range(N)]
        clients_indexes = [[0] * 100 for _ in range(N)]
        classes_by_index = [[] for _ in range(100)]
        classes_by_index_len = [0 for _ in range(100)]
        for i, label in enumerate(original_labels):
            for j in range(K):
                classes_by_index[label].append(i)
            classes_by_index_len[label] += K

        
        for i in range(N):
            # p = torch.tensor(classes_by_index_len) / sum(classes_by_index_len)
            p = torch.ones((len(classes_by_index_len), ))
            q = dirichlet.Dirichlet(0.1 * p).sample()
            # print(q)
            while(len(clients_labels[i]) < M):
                sampled_class = torch.multinomial(q, 1)
                if classes_by_index_len[sampled_class] == 0:
                    q = reweight(q, sampled_class)
                    # print(q)
                else:
                    sampled_index = random.randint(0, classes_by_index_len[sampled_class] - 1)
                    sampled_original_index = classes_by_index[sampled_class][sampled_index]
                    clients_images[i].append(original_images[sampled_original_index])
                    clients_labels[i].append(original_labels[sampled_original_index])
                    clients_indexes[i][sampled_class] += 1
                    classes_by_index[sampled_class].pop(sampled_index)
                    classes_by_index_len[sampled_class] -= 1
            clients_labels[i] = torch.tensor(clients_labels[i])
            clients_images[i] = torch.tensor([image for image in clients_images[i]])
        
        return clients_images, clients_labels, clients_indexes

    def get_iid_class_and_labels(self, original_images, original_labels, N):

        clients_images = [[] for _ in range(N)]
        clients_labels = [[] for _ in range(N)]
        M = len(original_labels) // N
        for i in range(N):
            for j in range(M):
                clients_images[i].append(original_images[i * M + j])
                clients_labels[i].append(original_labels[i * M + j])
        clients_labels[i] = torch.tensor(clients_labels[i])
        clients_images[i] = torch.tensor([image for image in clients_images[i]])


        
        
        return clients_images, clients_labels


    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        original_dataset_tr = CIFAR100(root, train=True, download=True, transform=transform_train)
        original_dataset_te = CIFAR100(root, train=False, download=True, transform=transform_test)
        

        # original_dataset_tr = CIFAR100(root, train=True, download=True, transform=transforms.ToTensor())
        # original_dataset_te = CIFAR100(root, train=False, download=True, transform=transforms.ToTensor())

        # original_images = torch.cat((original_dataset_tr.data,
        #                              original_dataset_te.data))

        # original_labels = torch.cat((original_dataset_tr.targets,
        #                              original_dataset_te.targets))

        original_images_tr = [X.numpy() for X, Y in original_dataset_tr]
        original_labels_tr = [Y for X, Y in original_dataset_tr]

        original_images_te = [X.numpy() for X, Y in original_dataset_te]
        original_labels_te = [Y for X, Y in original_dataset_te]



        original_datas = list(zip(original_images_tr, original_labels_tr))
        random.shuffle(original_datas)

        original_images_tr[:], original_labels_tr[:] = zip(*original_datas)

        # print(original_labels_tr)

        self.datasets = []

        # clients_images, clients_labels = self.get_iid_class_and_labels(original_images_tr, original_labels_tr, len(environments) - 10)

        clients_images, clients_labels, clients_indexes = self.get_noniid_class_and_labels(original_images_tr, original_labels_tr, len(environments) - 10)
        clients_images_te, clients_labels_te, _ = self.get_noniid_class_and_labels(original_images_te, original_labels_te, 10)

        print(clients_indexes)

        for i in range(len(environments)):
            if i < len(environments) - 10:
                # self.datasets.append(dataset_transform(torch.tensor(clients_images[i]), torch.tensor(clients_labels[i]), None))
                self.datasets.append(TensorDataset(torch.tensor(clients_images[i]), torch.tensor(clients_labels[i])))
                # self.datasets.append(TensorDataset(torch.tensor(original_images_tr), torch.tensor(original_labels_tr)))
            else:
                self.datasets.append(TensorDataset(torch.tensor(original_images_te), torch.tensor(original_labels_te)))

        # for i in range(10):
        #     self.datasets.append(dataset_transform(torch.tensor(clients_images_te[i]), torch.tensor(clients_labels_te[i]), environments[-i - 1]))



        # for i in range(len(environments)):
        #     images = original_images[i::len(environments)]
        #     labels = original_labels[i::len(environments)]
        #     self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class RotatedCIFAR100(MultipleEnvironmentCifar100):

    def __init__(self, root, train_envs, hparams):
        super(RotatedCIFAR100, self).__init__(root, [0] * train_envs + [0, 15, 30, 45, 60, 75, 90, 105, 120, 135],
                                           self.rotate_dataset, (3, 32, 32,), 100)

        # super(RotatedCIFAR10, self).__init__(root, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                    self.rotate_dataset, (3, 32, 32,), 10)

    def rotate_dataset(self, images, labels, angle):
        # rotation = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
        #         interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
        #     transforms.ToTensor()])
        if angle is None:
            angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
            p = torch.ones((10,)) / 10
            q = dirichlet.Dirichlet(1.0 * p).sample()
            print(q)

            x = torch.zeros(len(images), 3, 32, 32)
            for i in range(len(images)):
                angle = angles[torch.multinomial(q, 1)]
                angle = np.random.randint(0, 180)
                rotation = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Lambda(lambda x: rotate(x, angle)),
                    transforms.ToTensor()])
                x[i] = rotation(images[i])
        else:

            rotation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: rotate(x, angle)),
                transforms.ToTensor()])

            x = torch.zeros(len(images), 3, 32, 32)
            for i in range(len(images)):
                x[i] = rotation(images[i])


        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentCifar10(MultipleDomainDataset):

    def get_noniid_class_and_labels(self, original_images, original_labels, N):
    
        M = len(original_labels) // N

        clients_images = [[] for _ in range(N)]
        clients_labels = [[] for _ in range(N)]
        classes_by_index = [[] for _ in range(10)]
        classes_by_index_len = [0 for _ in range(10)]
        for i, label in enumerate(original_labels):
            classes_by_index[label].append(i)
            classes_by_index_len[label] += 1

        
        for i in range(N):
            p = torch.tensor(classes_by_index_len) / sum(classes_by_index_len)
            q = dirichlet.Dirichlet(0.1 * p).sample()
            while(len(clients_labels[i]) < M):
                sampled_class = torch.multinomial(q, 1)
                if classes_by_index_len[sampled_class] == 0:
                    q = reweight(q, sampled_class)
                    # print(q)
                else:
                    sampled_index = random.randint(0, classes_by_index_len[sampled_class] - 1)
                    sampled_original_index = classes_by_index[sampled_class][sampled_index]
                    clients_images[i].append(original_images[sampled_original_index])
                    clients_labels[i].append(original_labels[sampled_original_index])
                    classes_by_index[sampled_class].pop(sampled_index)
                    classes_by_index_len[sampled_class] -= 1
            clients_labels[i] = torch.tensor(clients_labels[i])
            clients_images[i] = torch.tensor([image for image in clients_images[i]])
        
        return clients_images, clients_labels

    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        mean = [0.48836562, 0.48134598, 0.4451678]
        std = [0.24833508, 0.24547848, 0.26617324]

        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            transforms.ToTensor()
        ])

        

        original_dataset_tr = CIFAR10(root, train=True, download=True, transform=transform_train)
        original_dataset_te = CIFAR10(root, train=False, download=True, transform=transform_test)

        # original_images = torch.cat((original_dataset_tr.data,
        #                              original_dataset_te.data))

        # original_labels = torch.cat((original_dataset_tr.targets,
        #                              original_dataset_te.targets))

        original_images_tr = [X.numpy() for X, Y in original_dataset_tr]
        original_labels_tr = [Y for X, Y in original_dataset_tr]

        original_images_te = [X.numpy() for X, Y in original_dataset_te]
        original_labels_te = [Y for X, Y in original_dataset_te]

        print(original_images_tr[0])


        original_datas = list(zip(original_images_tr, original_labels_tr))
        random.shuffle(original_datas)

        original_images_tr[:], original_labels_tr[:] = zip(*original_datas)

        print(original_images_tr[0])

        self.datasets = []

        clients_images, clients_labels = self.get_noniid_class_and_labels(original_images_tr, original_labels_tr, len(environments) - 10)
        clients_images_te, clients_labels_te = self.get_noniid_class_and_labels(original_images_te, original_labels_te, 10)

        for i in range(len(environments)):
            if i < len(environments) - 10:
                self.datasets.append(dataset_transform(torch.tensor(clients_images[i]), torch.tensor(clients_labels[i]), None))
                # self.datasets.append(TensorDataset(torch.tensor(clients_images[i]), torch.tensor(clients_labels[i])))
                # self.datasets.append(TensorDataset(torch.tensor(original_images_tr), torch.tensor(original_labels_tr)))
            else:
                self.datasets.append(dataset_transform(torch.tensor(original_images_te), torch.tensor(original_labels_te), environments[i]))
                # self.datasets.append(TensorDataset(torch.tensor(original_images_te), torch.tensor(original_labels_te)))



        # for i in range(len(environments)):
        #     images = original_images[i::len(environments)]
        #     labels = original_labels[i::len(environments)]
        #     self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class RotatedCIFAR10(MultipleEnvironmentCifar10):

    def __init__(self, root, train_envs, hparams):
        super(RotatedCIFAR10, self).__init__(root, [0] * train_envs + [0, 15, 30, 45, 60, 75, 90, 105, 120, 135],
                                           self.rotate_dataset, (3, 32, 32,), 10)

        # super(RotatedCIFAR10, self).__init__(root, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                    self.rotate_dataset, (3, 32, 32,), 10)

    def rotate_dataset(self, images, labels, angle):
        # rotation = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
        #         interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
        #     transforms.ToTensor()])
        if not angle:
            angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
            p = torch.ones((10,)) / 10
            q = dirichlet.Dirichlet(1.0 * p).sample()
            print(q)

            x = torch.zeros(len(images), 3, 32, 32)
            for i in range(len(images)):
                angle = angles[torch.multinomial(q, 1)]
                rotation = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Lambda(lambda x: rotate(x, angle)),
                    transforms.ToTensor()])
                x[i] = rotation(images[i])
        else:

            rotation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: rotate(x, angle)),
                transforms.ToTensor()])

            x = torch.zeros(len(images), 3, 32, 32)
            for i in range(len(images)):
                x[i] = rotation(images[i])


        y = labels.view(-1)

        return TensorDataset(x, y)


    



class MultipleEnvironmentMNIST(MultipleDomainDataset):

    def get_noniid_class_and_labels(self, original_images, original_labels, N):

        M = len(original_labels) // N

        clients_images = [[] for _ in range(N)]
        clients_labels = [[] for _ in range(N)]
        classes_by_index = [[] for _ in range(10)]
        classes_by_index_len = [0 for _ in range(10)]
        for i, label in enumerate(original_labels):
            classes_by_index[label].append(i)
            classes_by_index_len[label] += 1

        
        for i in range(N):
            p = torch.tensor(classes_by_index_len) / sum(classes_by_index_len)
            q = dirichlet.Dirichlet(0.1 * p).sample()
            while(len(clients_labels[i]) < M):
                sampled_class = torch.multinomial(q, 1)
                if classes_by_index_len[sampled_class] == 0:
                    q = reweight(q, sampled_class)
                    # print(q)
                else:
                    sampled_index = random.randint(0, classes_by_index_len[sampled_class] - 1)
                    sampled_original_index = classes_by_index[sampled_class][sampled_index]
                    clients_images[i].append(original_images[sampled_original_index])
                    clients_labels[i].append(original_labels[sampled_original_index])
                    classes_by_index[sampled_class].pop(sampled_index)
                    classes_by_index_len[sampled_class] -= 1
            clients_labels[i] = torch.tensor(clients_labels[i])
            clients_images[i] = torch.tensor([image.numpy() for image in clients_images[i]])
        
        return clients_images, clients_labels
            



        
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        # original_images = torch.cat((original_dataset_tr.data,
        #                              original_dataset_te.data))

        # original_labels = torch.cat((original_dataset_tr.targets,
        #                              original_dataset_te.targets))

        original_images_tr = original_dataset_tr.data
        original_labels_tr = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images_tr))

        original_images_tr = original_images_tr[shuffle]
        original_labels_tr = original_labels_tr[shuffle]

        self.datasets = []

        clients_images, clients_labels = self.get_noniid_class_and_labels(original_images_tr, original_labels_tr, 10)
        clients_images_te, clients_labels_te = self.get_noniid_class_and_labels(original_dataset_te.data, original_dataset_te.targets, len(environments) - 10)

        for i in range(len(environments)):
            if i < 10:
                self.datasets.append(dataset_transform(torch.tensor(clients_images[i]), torch.tensor(clients_labels[i]), environments[i]))
            else:
                self.datasets.append(dataset_transform(torch.tensor(original_dataset_te.data), torch.tensor(original_dataset_te.targets), environments[i]))

        for i in range(len(environments) - 10):
            self.datasets.append(dataset_transform(torch.tensor(clients_images_te[i]), torch.tensor(clients_labels_te[i]), environments[i + 10]))



        # for i in range(len(environments)):
        #     images = original_images[i::len(environments)]
        #     labels = original_labels[i::len(environments)]
        #     self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = 15

    def __init__(self, root, train_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9, 0.9, 0.1, 0.2, 0.5, 0.8, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)
        # super(ColoredMNIST, self).__init__(root, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.8, 0.9, 0.8, 0.9, 0.1, 0.2, 0.5, 0.8, 0.9],
                                        #  self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = 15

    def __init__(self, root, train_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        # rotation = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
        #         interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
        #     transforms.ToTensor()])
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,))),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):

    def split_datasets(self, dataset):
        class_num = len(dataset.classes)
        client_1_X = []
        client_1_Y = []
        client_2_X = []
        client_2_Y = []
        for X, Y in dataset:
            # print(X, Y)
            if Y <= class_num // 2:
                client_1_X.append(X.numpy())
                client_1_Y.append(Y)
            else:
                client_2_X.append(X.numpy())
                client_2_Y.append(Y)
        client_1_X = torch.tensor(client_1_X)
        client_1_Y = torch.tensor(client_1_Y)
        client_2_X = torch.tensor(client_2_X)
        client_2_Y = torch.tensor(client_2_Y)
        return [TensorDataset(client_1_X, client_1_Y), TensorDataset(client_2_X, client_2_Y)]

    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            # if augment and (i not in test_envs):
            #     env_transform = augment_transform
            # else:
            #     env_transform = transform

            env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.num_classes = len(env_dataset.classes)

            clients = self.split_datasets(env_dataset)

            self.datasets.append(clients[0])
            self.datasets.append(clients[1])

        self.input_shape = (3, 32, 32,)
        # self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

