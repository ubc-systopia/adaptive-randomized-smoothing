import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.distributions as D

from models import architectures
from utils import get_image_size

class SINGLE_QUERY_ARCH(nn.Module):
    def __init__(self, conf):
        super(SINGLE_QUERY_ARCH, self).__init__()

        self.conf = conf
        if "cuda" in str(self.conf.trainer.device): 
            self.device_string = self.conf.trainer.device
        else:
            self.device_string = f'cuda:{self.conf.trainer.device}'
        self.image_size = get_image_size(conf)
        self.image_dims = self.image_size * self.image_size * 3

        # if learning an average mask in first query, initialize a random or identity mask
        if conf.arch.mask_type == "static":
            mask_shape = (self.image_size, self.image_size)
            if conf.arch.mask_init == "random":
                self.static_mask = nn.Parameter(torch.rand(mask_shape), requires_grad=True)
            elif conf.arch.mask_init == "identity":
                self.static_mask = nn.Parameter(torch.ones(mask_shape), requires_grad=True)
            
        # finally, initialize base classifier (common to vanilla and adaptive mode)
        self.base_classifier = architectures.get_architecture(conf,
                                                              arch=conf.arch.base_classifier,
                                                              prepend_preprocess_layer=True,
                                                              prepend_normalize_layer=True,
                                                              dataset=conf.data.dataset,
                                                              input_size=self.image_size,
                                                              num_classes=conf.data.num_classes)

    def forward(self, x, logging_trackers):
        x_original = copy.deepcopy(x)

        # FIRST QUERY
        if self.conf.arch.mask_type == "static":
            x = torch.mul(x, self.static_mask)
            norm_1 = torch.sqrt(torch.tensor(3)) * self.static_mask.norm(2)
            # computing sigma for first query based on mask's l2 norm
            sigma_1 = (self.conf.arch.total_sigma) * (norm_1 / np.sqrt(self.image_dims))
        elif self.conf.arch.mask_type == "vanilla":
            # -- if no first query mask, then add noise directly to inputs (without transformation)
            # -- in the second term below, numerator and denominator are same because no transformation
            sigma_1 = self.conf.arch.total_sigma
            
        norm_dist = D.Normal(loc=0., scale=sigma_1)
        noise = norm_dist.rsample(x.shape).to(self.device_string)
        x += noise # adding the sampled noise

        # Perform the usual forward pass; passing the second query's noisy image
        output_pred = self.base_classifier(x)

        # save intermediate images 
        if logging_trackers["epoch"] % self.conf.visualize.save_fig_epoch == 0 or logging_trackers["epoch"] == 1:  
            if logging_trackers["batch_idx"] == self.conf.visualize.which_batch:
                if self.conf.visualize.save_intermediate_imgs:
                    num_to_store = self.conf.visualize.num_to_store
                    torch.save(x_original[:num_to_store], os.path.join(logging_trackers["saved_dir"], "original_images.pt"))
                    torch.save(x[:num_to_store], os.path.join(logging_trackers["saved_dir"], "q_1_images.pt"))
                    if self.conf.arch.mask_type == "static":
                        torch.save(self.static_mask, os.path.join(logging_trackers["saved_dir"], "static_mask.pt"))
                if self.conf.visualize.save_intermediate_fig: 
                    raise RuntimeError("saving pipeline figures for cohen or static training is not implemented, but is implemented in two_query_arch.py for ars") 
            
        return output_pred
