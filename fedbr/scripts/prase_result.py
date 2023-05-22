import argparse
import collections
from copy import deepcopy
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
from parso import parse
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
from torchvision.datasets import MNIST, ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import rotate
from torch.utils.data import TensorDataset, Subset
from fedbr.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from fedbr.lib import misc

def rotate_dataset(images, labels, angle):
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


original_dataset_te = MNIST('./fedbr/data/MNIST/', train=False, download=True)
datasets = []
envs = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
for env in envs:
    datasets.append(rotate_dataset(torch.tensor(original_dataset_te.data), torch.tensor(original_dataset_te.targets), env))

results = {}
eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64, num_workers=8) for env in datasets]
eval_loader_names = ['env{0}_in'.format(str(i).zfill(2))
        for i in range(len(datasets))]



original_dataset = 'RotatedMNIST'
device = 0

algorithm = 'ERM'
dataset = 'Rotate_mnist'
k = 50
dir = f'train_output_{algorithm}_{dataset}_{k}/'
evals = zip(eval_loader_names, eval_loaders, None)
for step in range(0, 50000, 100):
    model = torch.load(dir + f'model_step{step}_{algorithm}_{original_dataset}_{k}.pkl')
    for name, loader, weights in evals:
        acc = misc.accuracy(model, loader, weights, device)
        results[name+'_acc'] = acc
    results_keys = sorted(results.keys())
    if results_keys != last_results_keys:
        misc.print_row(results_keys, colwidth=12)
        last_results_keys = results_keys
    misc.print_row([results[key] for key in results_keys],
        colwidth=12)

