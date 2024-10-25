import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.distributions as D

from src.models import architectures
from src.utils import get_image_size, convert_image_np


class TWO_QUERY_ARCH_DDP(nn.Module):
    def __init__(self, conf, args, current_device):
        super(TWO_QUERY_ARCH_DDP, self).__init__()

        self.conf = conf
        if "cuda" not in str(self.conf.trainer.device): 
            raise ValueError("need cuda for ddp settings")
        else:
            self.current_device = current_device
            self.device_string = torch.device('cuda:'+ str(current_device))
        self.image_size = get_image_size(conf)
        self.image_dims = self.image_size * self.image_size * 3

        # initialize second query mask model
        self.mask_model = architectures.get_architecture(conf=conf,
                                                         arch=conf.arch.mask_model,
                                                         prepend_preprocess_layer=False,
                                                         prepend_normalize_layer=False)

        
        # budget splitting mechanism, using fixed split
        self.first_query_budget_frac = torch.tensor(np.sqrt(conf.arch.first_query_budget_frac), device=self.device_string)
            
        # initialize base classifier
        self.base_classifier = architectures.get_architecture(conf=conf,
                                                              arch=conf.arch.base_classifier,
                                                              prepend_preprocess_layer=True,
                                                              prepend_normalize_layer=True,
                                                              dataset=conf.data.dataset,
                                                              input_size=self.image_size,
                                                              num_classes=conf.data.num_classes)
        if args.resume_mask_only is None:
            pass
        else :
            print("load pretrained classifier from {}, and start mask only training".format(args.resume_mask_only))
            checkpoint = torch.load(args.resume_mask_only)
            clean_key={key.lstrip('base_classifier.'):value for key, value in checkpoint.items()} #remove 'base_classifier.' in keys of checkpint if any, otw keys do not match with default loaded structure
            self.base_classifier.load_state_dict(clean_key)
            self.base_classifier.eval()

    def forward(self, x, logging_trackers: dict = {}):
        x_original = copy.deepcopy(x)

        # FIRST QUERY
        # doing ars training, add noise directly to inputs (without transformation)
        sigma_1 = self.conf.arch.total_sigma / self.first_query_budget_frac
        norm_dist = D.Normal(loc=0., scale=sigma_1)
        noise = norm_dist.rsample(x.shape).to(self.device_string)
        x += noise 

        # SECOND QUERY BEGINS
        # - pass the noisy image from first query into second query mask model
        # - get the per-input mask and
        # - multiply original images with those corresponding masks 
        # clamped_second_query_masks = self.mask_model(x)
        if self.conf.trainer.mask_recon > 0 : # do supervised training with original image reconstruction
            clamped_second_query_masks_with_recon = self.mask_model(x).to(torch.float32)
            assert clamped_second_query_masks_with_recon.shape[1] == 4
            clamped_second_query_masks = clamped_second_query_masks_with_recon[:,0:1,:,:] # to get only the mask of the output, [:,1:4,:,:] is the reconstruction
        else: 
        # print("using no reconstructioin!") # do unsupervised ars
            clamped_second_query_masks= self.mask_model(x).to(torch.float32)
            assert clamped_second_query_masks.shape[1] == 1


        # multiply raw input image with the obtained masks
        x_transformed = torch.mul(x_original, clamped_second_query_masks)
        # compute norm from mask
        norm_2 = torch.sqrt(torch.tensor(3)) * torch.norm(clamped_second_query_masks.view(clamped_second_query_masks.shape[0], -1), p=2, dim=1)
        # computing the budget for second query (according to GDP formulation)
        second_query_budget_frac = torch.sqrt(1 - torch.square(self.first_query_budget_frac)).to(self.device_string)
        # compute sigma from norms and query budget
        sigma_2 = (self.conf.arch.total_sigma / second_query_budget_frac) * (norm_2 / np.sqrt(self.image_dims))
        # add sigma bound, otw sigma will occasionaly has 0 values thus sampling from normal fails
        sigma_2 = torch.maximum(torch.ones(sigma_2.shape).to(self.device_string) * 1e-6, sigma_2)  
        # adding noise (parallely)
        norm_dist = D.Normal(loc=0., scale=sigma_2)
        noise = norm_dist.rsample(x_transformed.shape[1:]).permute(3,0,1,2).to(self.device_string)
        x_transformed += noise

        # averaging without lambda_1 * clamped_second_query_masks)
        pre_averaging_x_transformed = x_transformed.clone().detach() # used if want to save figures 
        # computing the weights
        sigma_1 = sigma_1.repeat(sigma_2.shape[0]).reshape(sigma_2.shape[0],1,1,1).repeat(1,1,self.image_size,self.image_size)
        sigma_2 = sigma_2.reshape(sigma_2.shape[0],1,1,1).repeat(1,1,self.image_size,self.image_size)
        denominator = ((clamped_second_query_masks ** 2) * (sigma_1 ** 2)) + (sigma_2 ** 2)
        w1 = sigma_2 ** 2 / denominator
        w2 = ((sigma_1 ** 2) * (clamped_second_query_masks)) / denominator
        # average noisy images
        x_transformed *= w2
        x_transformed += (w1 * x)

        # Perform the usual forward pass (passing the second query's noisy image)
        output_pred = self.base_classifier(x_transformed)

        # save intermediate images 
        if self.current_device == 0:
            if logging_trackers["epoch"] % self.conf.visualize.save_fig_epoch == 0 or logging_trackers["epoch"] == 1:
                if logging_trackers["batch_idx"] == self.conf.visualize.which_batch: #batch_idx here is the related to (but not always equal to) class label, controls which image you will save
                    # to save more intermediate images for future use if you want
                    if self.conf.visualize.save_intermediate_imgs:
                        num_to_store = self.conf.visualize.num_to_store
                        torch.save(x_original[:num_to_store], os.path.join(logging_trackers["saved_dir"], "original_images.pt"))
                        torch.save(x[:num_to_store], os.path.join(logging_trackers["saved_dir"], "q_1_images.pt"))
                        torch.save(clamped_second_query_masks[:num_to_store], os.path.join(logging_trackers["saved_dir"], "post_clamped_second_query_masks.pt"))
                        torch.save(pre_averaging_x_transformed[:num_to_store], os.path.join(logging_trackers["saved_dir"], "q_2_images_before_averaging.pt"))
                        torch.save(x_transformed[:num_to_store], os.path.join(logging_trackers["saved_dir"], "q_2_images_after_averaging.pt"))

                    # this pipline figure contains 6 images per input image, which corresponds to 6 images in Figure 1 in the paper.
                    if self.conf.visualize.save_intermediate_fig: 
                        indices = np.array([0,10,20,30,40])
                        f, axarr = plt.subplots(6, 1, figsize=(30,30))
                        original_images = convert_image_np(torchvision.utils.make_grid(x_original[indices], nrow=len(indices))) 
                        q1_noisy_images = convert_image_np(torchvision.utils.make_grid(x[indices], nrow=len(indices)))
                        masks = convert_image_np(torchvision.utils.make_grid(clamped_second_query_masks[indices], nrow=len(indices)))[:, :, 0]
                        masked_images = torch.mul(x_original, clamped_second_query_masks)
                        masked_images = convert_image_np(torchvision.utils.make_grid(masked_images[indices], nrow=len(indices)))
                        q2_images_before_avg = convert_image_np(torchvision.utils.make_grid(pre_averaging_x_transformed[indices], nrow=len(indices)))
                        q2_images_after_avg = convert_image_np(torchvision.utils.make_grid(x_transformed[indices], nrow=len(indices)))
                        axarr[0].imshow(original_images)
                        axarr[1].imshow(q1_noisy_images)
                        axarr[2].imshow(masks, cmap="gray")
                        axarr[3].imshow(masked_images)
                        axarr[4].imshow(q2_images_before_avg)
                        axarr[5].imshow(q2_images_after_avg)
                        plt.tight_layout()
                        f.savefig(os.path.join(logging_trackers["saved_dir"], "pipeline.png"))

        if self.conf.trainer.mask_supervise > 0:
            shape = clamped_second_query_masks.shape
            return output_pred, clamped_second_query_masks.view(shape[0], shape[2], shape[3])
        
        return output_pred
