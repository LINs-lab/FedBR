import logging
import argparse
from copy import deepcopy
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import utils
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm


import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from style_GAN_v2 import StyleGAN2



sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


from distribution_utils import train_distribution_diversity
# from loss_fn.cov_loss import (
#     cov_non_diag_norm, cov_norm
# )


# from build import create_model



import time

def check_device(data_src, device=None):
    if device is not None:
        if data_src.device is not device:
            return data_src.to(device)
        else:
            return data_src
    else:
        return data_src

def cov_non_diag_norm(x, size=[8, 8]):
    # Calculate Covariance Matrix
    cov_sum = 0
    b, c, w, h = x.size()
    # print(f"x.size(): {x.size()}")
    temp_matrix = 1 - torch.eye(int(size[0]*size[1]), dtype=torch.float, requires_grad=False)

    temp_matrix = check_device(temp_matrix, x)
    #     temp_matrix = 1 - torch.eye(22*3, dtype=torch.FloatTensor, require_grad=False)
    down_x = F.interpolate(x, size=size, mode="bilinear")
    # print(f"down_x.shape: {down_x.shape}")
    for i in range(c):
        sub_x = down_x[:, i, :, :]
        # print(f"sub_x.shape: {sub_x.shape}")
        #         down_x = F.interpolate(sub_x, size=size)
        sub_x = sub_x.view(sub_x.size(0), -1)
        # sub_x = sub_x / torch.linalg.norm(sub_x, dim=1, keepdim=True)
        sub_x = sub_x / sub_x.norm(dim=1, keepdim=True)
        cov = torch.mm(sub_x.t(), sub_x)**2 / (b - 1)
        # print(f"cov.shape: {cov.shape}")
        #         cov = torch.sum(torch.bmm(down_x, down_x.transpose(0, 2, 1)), dim=0) / (x.size(0) - 1)
        # print(f"temp_matrix.shape: {temp_matrix.shape}")
        cov_sum += torch.mean(cov * temp_matrix)
    
    return cov_sum

# def make_noise():
#     device = self.input.input.device

#     noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

#     for i in range(3, self.log_size + 1):
#         for _ in range(2):
#             noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    return noises

def mean_latent(style, n_latent, device):
    latent_in = torch.randn(
        n_latent, style.style_dim, device=device
    )
    latent = style(latent_in).mean(0, keepdim=True)

    return latent

def generate(args, device):

    # g_ema = create_model(args, args.model, output_dim=10).to(device)
    net = StyleGAN2(32)
    style = net.S.to(device)
    generator = net.G.to(device)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = mean_latent(style, args.truncation_mean, device)
    else:
        mean_latent = None

    noise_num = args.noise_num

    n_dim = generator.num_layers
    normed_n_mean = train_distribution_diversity(
        n_distribution=noise_num, n_dim=n_dim, max_iters=500)
    style_GAN_latent_noise_mean = normed_n_mean.detach().to(device)
    style_GAN_latent_noise_std = [0.1 / n_dim] * n_dim

    global_zeros = torch.ones((noise_num, generator.latent_dim)) * 0.0
    global_mean_vector = torch.normal(mean=global_zeros, std=args.style_gan_sample_z_mean)
    style_GAN_sample_z_mean = global_mean_vector
    style_GAN_sample_z_std = args.style_gan_sample_z_std

    net.eval()

    # TODO, for more efficiently loading data.
    # if args.package:

    with torch.no_grad():
        for noise_i in tqdm(range(args.noise_num)):
            iters = args.sample // args.batch_size

            train = "train"
            class_dir = f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}"

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            for batch_i in tqdm(range(iters)):
                mean_vector = style_GAN_sample_z_mean[noise_i].repeat(n_dim, args.batch_size,1)
                sample_z = torch.normal(mean=mean_vector, std=style_GAN_sample_z_std).to(device)

                # latent_noise_mean_i = style_GAN_latent_noise_mean[noise_i]

                noise_mean_vector = style_GAN_latent_noise_mean[noise_i]

                sample_noise = list(zip(noise_mean_vector, style_GAN_latent_noise_std))

                # print(sample_z.shape)
                sample = generator(
                    sample_z,
                    sample_noise
                )
                trans = torchvision.transforms.Resize((args.image_resolution, args.image_resolution))
                if args.image_resolution == 28:
                    sample = torch.mean(sample, 1).unsqueeze(1)
                sample = trans(sample)
                print(f"sample.shape: {sample.shape}")
                for j in range(args.batch_size):
                    index = batch_i * args.batch_size + j 
                    utils.save_image(
                        sample[j],
                        f"{class_dir}/{str(index).zfill(6)}.jpg"
                    )
            utils.save_image(
                sample,
                f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}-overview.jpg",
                nrow=args.overview_n_columns
            )


def train_generator_diversity(device, generator, max_iters=100, min_loss=0.0):
    generator.train()
    generator.to(device)
    for i in range(max_iters):
        generator_optimizer = torch.optim.SGD(generator.parameters(),
            lr=0.01, weight_decay=0.0001, momentum=0.9)
        means = torch.zeros((64, args.vae_decoder_z_dim))
        z = torch.normal(mean=means, std=1.0).to(device)
        data = generator(z)
        loss_diverse = cov_non_diag_norm(data)
        generator_optimizer.zero_grad()
        loss_diverse.backward()
        generator_optimizer.step()
        print(f"Iteration: {i}, loss_diverse: {loss_diverse.item()}")
        if loss_diverse.item() < min_loss:
            print(f"Iteration: {i}, loss_diverse: {loss_diverse.item()} smaller than min_loss: {min_loss}, break")
            break
    generator.cpu()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the generator")


    parser.add_argument(
        "--model", type=str, default="style_GAN_v2_G", help="output image size of the generator"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", help="Original dataset"
    )
    parser.add_argument(
        "--generate_dataset", type=str, default="style_GAN_init", help="output image size of the generator"
    )
    parser.add_argument(
        "--gpu_index", type=int, default=0, help="output image size of the generator"
    )
    parser.add_argument(
        "--fedaux", type=bool, default=False, help="output image size of the generator"
    )
    parser.add_argument(
        "--VHL", type=bool, default=True, help="output image size of the generator"
    )
    parser.add_argument(
        "--VHL_label_style", type=bool, default=True, help=""
    )
    parser.add_argument(
        "--fed_moon", type=bool, default=False, help=""
    )
    parser.add_argument(
        "--gate_layer", type=bool, default=False, help=""
    )


    parser.add_argument(
        "--root_path", type=str, default="./dataset", help="output image size of the generator"
    )

    parser.add_argument(
        "--image_resolution", type=int, default=32, help="output image size of the generator"
    )
    parser.add_argument(
        "--noise_num", type=int, default=10, help=""
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help=""
    )
    parser.add_argument(
        "--overview_n_columns", type=int, default=10, help=""
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="number of samples to be generated for each label",
    )
    parser.add_argument(
        "--style_gan_style_dim", type=int, default=64, help=""
    )
    parser.add_argument(
        "--style_gan_n_mlp", type=int, default=1, help=""
    )
    parser.add_argument("--package", type=bool, default=True, help="")
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--style_gan_cmul",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--style_gan_sample_z_mean",
        type=float,
        default=0.5,
        help="",
    )
    parser.add_argument(
        "--style_gan_sample_z_std",
        type=float,
        default=0.3,
        help="",
    )
    parser.add_argument(
        "--vae_decoder_z_dim",
        type=int,
        default=8,
        help="",
    )
    parser.add_argument(
        "--vae_decoder_ngf",
        type=int,
        default=64,
        help="",
    )



    args = parser.parse_args()

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    time_table = {}
    time_now = time.time()
    print("Creating model")

    # g_ema.load_state_dict(checkpoint["g_ema"])

    print("generating images")
    # if args.model == "cifar_conv_decoder":
    #     generate_from_vae_decoder(args, device)
    generate(args, device)

    time_table['generate_images'] = time.time() - time_now
    time_now = time.time()
    print(time_table)











