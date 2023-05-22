import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from fedbr.algorithms import Algorithm

class Algorithm_Fed():

    def __init__(self, algorithm):

        self.algorithm = algorithm
        self.original = copy.deepcopy(self.algorithm)
        self.gradients = None

    def get_gradients(self):
        old_dict = copy.deepcopy(self.original.state_dict())
        new_dict = self.algorithm.state_dict()
        for k, v in new_dict.items():
            old_dict[k] = 1.0 * (old_dict[k] - new_dict[k])
        return old_dict

    def get_old_new_gradients(self, old, new):
        grads = []
        old_dict = list(old.parameters())
        new_dict = list(new.parameters())
        for i in range(len(new_dict)):
            grads.append(1.0 * (old_dict[i] - new_dict[i]))
        return grads



    def predict(self, x):
        return self.algorithm.predict(x)

    def eval(self):
        self.algorithm.eval()

    def train(self):
        self.algorithm.train()

    def state_dict(self):
        return self.algorithm.state_dict()

    def load_state_dict(self, dict):

        return self.algorithm.load_state_dict(dict)


    def update(self, minibatches, unlabeled=None):
        self.original = copy.deepcopy(self.algorithm)

        step_vals = self.algorithm.update(minibatches, unlabeled)


        if not self.gradients:
            self.gradients = self.get_gradients()
        else:
            new_gradients = self.get_gradients()
            for k, v in new_gradients.items():
                self.gradients[k] = self.gradients[k] + new_gradients[k]
        return step_vals
        

    def global_update(self, gradients):

        raise NotImplementedError

    def copy_from(self, algorithm):

        raise NotImplementedError


class FedAvg(Algorithm_Fed):

    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.p = 1.0

    def global_update(self, gradients, weights, params):
        
        old_state = copy.deepcopy(self.algorithm.state_dict())
        for k, v in old_state.items():
            for gradient_i, gradient in enumerate(gradients):
                old_state[k] = old_state[k] - (params['global_lr'] * gradient[k] * weights[gradient_i])
        
        self.algorithm.load_state_dict(old_state)

    # def global_update(self, algorithms):
    #     old_state = copy.deepcopy(algorithms[0].state_dict())
    #     for k, v in old_state.items():
    #         old_state[k] = old_state[k] / len(algorithms)
    #     for i in range(1, len(algorithms)):
    #         new_state = algorithms[i].state_dict()
    #         for k, v in old_state.items():
    #             old_state[k] += new_state[k] / len(algorithms) * 1.0
    #     self.algorithm.load_state_dict(old_state)

    def copy_from(self, algorithm):
        
        t = self.algorithm.update_count
        self.algorithm = copy.deepcopy(algorithm.algorithm)
        self.algorithm.update_count = t
        # self.algorithm.load_state_dict(copy.deepcopy(algorithm.algorithm.state_dict()))
        # self.algorithm.re_init_optimizer()
        self.gradients = None




class FedGroupDRO(Algorithm_Fed):

    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.p = 1.0
        self.tau = algorithm.hparams['groupdro_eta']

    def update(self, minibatches, unlabeled=None):
        self.algorithm.q = torch.ones(len(minibatches)).to(minibatches[0][0].device)
        self.original = copy.deepcopy(self.algorithm)

        step_vals = self.algorithm.update(minibatches, unlabeled)

        if not self.gradients:
            self.gradients = self.get_gradients()
            self.p = self.p * np.exp((self.tau * step_vals['loss']))
        else:
            new_gradients = self.get_gradients()
            for k, v in new_gradients.items():
                self.gradients[k] = self.gradients[k] + new_gradients[k]
        return step_vals

    def global_update(self, gradients, weights, params):
        
        old_state = self.algorithm.state_dict()
        for k, v in old_state.items():
            for gradient_i, gradient in enumerate(gradients):
                old_state[k] = old_state[k] - params['global_lr'] * gradient[k] * weights[gradient_i]
        
        self.algorithm.load_state_dict(old_state)

    def copy_from(self, algorithm):
        
        self.algorithm.load_state_dict(copy.deepcopy(algorithm.algorithm.state_dict()))
        self.gradients = None
