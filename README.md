# FedBR: Improving Federated Learning on Heterogeneous Data via Local Learning Bias Reduction

Official implementation of the ICML 2023 paper "[FedBR: Improving Federated Learning on Heterogeneous Data via Local Learning Bias Reduction](https://arxiv.org/abs/2205.13462)".

## Overview 
**Abstract:**
Federated Learning (FL) is a way for machines to learn from data that is kept locally, in order to protect the privacy of clients. This is typically done using local SGD, which helps to improve communication efficiency. However, such a scheme is currently constrained by slow and unstable convergence due to the variety of data on different clients' devices. In this work, we identify three under-explored phenomena of biased local learning that may explain these challenges caused by local updates in supervised FL. As a remedy, we propose FedBR, a novel unified algorithm that reduces the local learning bias on features and classifiers to tackle these challenges. FedBR has two components. The first component helps to reduce bias in local classifiers by balancing the output of the models. The second component helps to learn local features that are similar to global features, but different from those learned from other data sources. We conducted several experiments to test FedBR and found that it consistently outperforms other SOTA FL methods. Both of its components also individually show performance gains.



## How to run

### Requirments
To run the code in this repository, be sure to install the following packages:
```
numpy==1.20.3
wilds==1.2.2
imageio==2.9.0
gdown==3.13.0
torchvision==0.8.2
torch==1.7.1
tqdm==4.62.2
backpack==0.1
parameterized==0.8.1
Pillow==8.3.2
```

### A quick start
The code is built on the top of [DomainBed](https://github.com/facebookresearch/DomainBed). You can find all algorithm implementations in `algorithms.py`. 
By default, FedBR generates 32 pseudo-data (using Mixture) at the start of the training.

Note: Before running VHL, you need to first execute `generative/generate.sh` in order to generate virtual data. You can modify the size of the virtual data and the dataset name for different datasets. 


Here is an example of the script:

```bash
python3 -m fedbr.scripts.train_fed \
  --data_dir=./fedbr/data/CIFAR10/ \
  --algorithm ERM \
  --dataset RotatedCIFAR10 \
  --train_envs 10 \
  --steps 50000 \
  --output_dir output-cifar10/ERM \
  --local_steps 50 \
  --device 6 \
  --save_model_every_checkpoint \
  --seed 12345
```



## Bibliography
If you find this repository helpful for your project, please consider citing:
```
@inproceedings{guo2023fedbr,
  title={FedBR: Improving Federated Learning on Heterogeneous Data via Local Learning Bias Reduction},
  author={Guo, Yongxin, Tang, Xiaoying, and Lin, Tao},
  booktitle = {International Conference on Machine Learning (ICML)},
  year={2023}
}
```
