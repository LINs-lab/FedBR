import math
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        self.device = features[0].device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast}")
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast.mean()}")
        # logging.info(f"In SupCon, anchor_dot_contrast.device: {anchor_dot_contrast.device}, self.device: {self.device}")


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # logging.info(f"In SupCon, exp_logits.shape: {exp_logits.shape}, exp_logits: {exp_logits.mean()}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # if torch.any(torch.isnan(log_prob)):
        #     log_prob[torch.isnan(log_prob)] = 0.0
        logging.info(f"In SupCon, log_prob.shape: {log_prob.shape}, log_prob: {log_prob.mean()}")

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        # loss[torch.isnan(loss)] = 0.0
        if torch.any(torch.isnan(loss)):
            # loss[torch.isnan(loss)] = 0.0
            logging.info(f"In SupCon, features.shape: {features.shape}, loss: {loss}")
            raise RuntimeError
        loss = loss.view(anchor_count, batch_size).mean()


        return loss

class proxy_align_loss(nn.Module):
    def __init__(self,
            inter_domain_mapping=False,
            inter_domain_class_match=True,
            noise_feat_detach=False,
            noise_contrastive=False,
            inter_domain_mapping_matrix=None,
            inter_domain_weight=0.0, inter_class_weight=1.0,
            noise_supcon_weight=0.1,
            noise_label_shift=10, device=None):
        super(proxy_align_loss, self).__init__()

        self.inter_domain_mapping = inter_domain_mapping
        self.noise_feat_detach = noise_feat_detach
        self.noise_contrastive = noise_contrastive
        self.inter_domain_class_match = inter_domain_class_match
        self.inter_domain_mapping_matrix = inter_domain_mapping_matrix
        self.inter_domain_weight = inter_domain_weight
        self.inter_class_weight = inter_class_weight
        self.noise_supcon_weight = noise_supcon_weight
        self.noise_label_shift = noise_label_shift
        self.device = device
        self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)


    def forward(self, features, labels, real_batch_size):

        self.device = features[0].device

        time_table = {}
        time_now = time.time()
        # softmax_out = F.softmax(features, dim=1)

        # real_features = softmax_out[:real_batch_size]
        # noise_features = softmax_out[real_batch_size:]

        real_features = features[:real_batch_size]
        noise_features = features[real_batch_size:]

        noise_batch_size = noise_features.shape[0]

        # yonggang
        # if self.feat_detach:
        #     new_features = torch.cat([real_features, noise_features.clone().detach()], dim=0)
        # else:
        #     new_features = features
        # yonggang
        
        if self.inter_domain_mapping:
            noise_features = torch.matmul(noise_features, self.inter_domain_mapping_matrix.to(self.device))
            new_features = torch.cat([real_features, noise_features], dim=0)
        elif self.noise_feat_detach:
            new_features = torch.cat([real_features, noise_features.clone().detach()], dim=0)
        else:
            new_features = features

        # logging.debug(f"real_features.shape: {real_features.shape}, noise_features.shape:{noise_features.shape},\
        #     features.shape: {features.shape}, real_batch_size:{real_batch_size} ")
        # Here the noise_features[:real_batch_size] is designed in order to avoid overflow.

        if real_batch_size > noise_batch_size:
            align_domain_loss = torch.linalg.norm(real_features[:noise_batch_size] - noise_features, ord=2, dim=1).sum() \
                / float(real_batch_size)
        else:
            align_domain_loss = torch.linalg.norm(real_features - noise_features[:real_batch_size], ord=2, dim=1).sum() \
                / float(real_batch_size)
        time_table["align_domain_loss"] = time.time() - time_now
        time_now = time.time()

        # real_labels = labels[:real_batch_size]
        # noise_labels = labels[real_batch_size:] - self.noise_label_shift
        # align_cls_loss = cross_pair_norm(real_labels, real_features, noise_labels, noise_features)

        new_features = F.normalize(new_features, dim=1).unsqueeze(1)
        new_noise_features = F.normalize(noise_features, dim=1).unsqueeze(1)

        if self.inter_domain_class_match:
            real_labels = labels[:real_batch_size]
            noise_labels = labels[real_batch_size:] - self.noise_label_shift
            align_cls_loss = self.supcon_loss(new_features, labels=torch.cat([real_labels, noise_labels], dim=0), temperature=0.07, mask=None)
        else:
            align_cls_loss = self.supcon_loss(new_features, labels=labels, temperature=0.07, mask=None)

        if self.noise_contrastive:
            noise_labels = labels[real_batch_size:] - self.noise_label_shift
            noise_cls_loss = self.supcon_loss(new_noise_features, labels=noise_labels, temperature=0.07, mask=None)
            noise_cls_loss_value = noise_cls_loss.item()
        else:
            noise_cls_loss = 0.0
            noise_cls_loss_value = 0.0

        time_table["align_cls_loss"] = time.time() - time_now
        time_now = time.time()

        # logging.debug(f"Calculating proxy align loss, time: {time_table}")
        return self.inter_domain_weight * align_domain_loss + \
            self.inter_class_weight * align_cls_loss + self.noise_supcon_weight * noise_cls_loss, align_domain_loss.item(), align_cls_loss.item(), noise_cls_loss_value
