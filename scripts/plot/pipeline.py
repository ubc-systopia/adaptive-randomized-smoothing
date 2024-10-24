import os
import torch 
import torchvision 
import matplotlib.pyplot as plt
import numpy as np
import argparse

"""
sample commands
python scripts/plot/pipeline.py --total_sigma 0.5 --indices deterministic --log_dir "logs/"
"""

# matplotlibrc params to set for better, bigger, clear plots
SMALLER_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# take in the directory path of numpy array files
parser = argparse.ArgumentParser(description='Plot some plots')
parser.add_argument('--total_sigma', type=float, default="None", help='total sigma value for the plots')
parser.add_argument('--indices', type=str, help='random or deterministic')
parser.add_argument('--log_dir', type=str, help='root path to pick up all pt files from and also store the plot')
args = parser.parse_args()

# process images
def convert_image_np(tens):
    return tens.data.cpu().numpy().transpose((1, 2, 0))

# choose 5 random indices
if args.indices == "random":
    indices = np.random.choice(100, size=5) # 100 should be smaller than num_to_store in the yaml file you used
elif args.indices == "deterministic": #you can make your own choice
    indices = np.array([0,10,20,30,40])

#### PLOT ORIGINAL IMAGE

# Plot the results side-by-side
f, axarr = plt.subplots(6, 1, figsize=(30,30))

#### PLOT ORIGINAL IMAGES ####
original_images = torch.load(os.path.join(args.log_dir, "original_images.pt"))
original_images_grid = convert_image_np(torchvision.utils.make_grid(original_images[indices], nrow=len(indices)))     
axarr[0].imshow(original_images_grid)
axarr[0].set_title('original images', fontsize=50)
axarr[0].axis('off')

#### PLOT Q1 NOISY IMAGES ####
q_1_noisy_images = torch.load(os.path.join(args.log_dir, "q_1_images.pt"))
q_1_noisy_images_grid = convert_image_np(torchvision.utils.make_grid(q_1_noisy_images[indices], nrow=len(indices)))
axarr[1].imshow(q_1_noisy_images_grid)
axarr[1].set_title('1st query noisy images', fontsize=50)
axarr[1].axis('off')

#### PLOT MASKS ####
second_query_masks = torch.load(os.path.join(args.log_dir, "post_clamped_second_query_masks.pt"))
second_query_masks_grid = convert_image_np(torchvision.utils.make_grid(second_query_masks[indices], nrow=len(indices)))
mask_channel = second_query_masks_grid[:, :, 0]
axarr[2].imshow(mask_channel, cmap="gray")
axarr[2].set_title('Masks', fontsize=50)
axarr[2].axis('off')

#### PLOT MASKED IMAGES (multiplication happens here coz I didnt store the masked images) ####
masked_images = torch.mul(original_images, second_query_masks)
masked_images_grid = convert_image_np(torchvision.utils.make_grid(masked_images[indices], nrow=len(indices)))
axarr[3].imshow(masked_images_grid)
axarr[3].set_title('Masked images', fontsize=50)
axarr[3].axis('off')

#### PLOT Q2 IMAGES BEFORE AVERAGING ####
q_2_images_before_averaging = torch.load(os.path.join(args.log_dir, "q_2_images_before_averaging.pt"))
q_2_images_before_averaging_grid = convert_image_np(torchvision.utils.make_grid(q_2_images_before_averaging[indices], nrow=len(indices)))     
axarr[4].imshow(q_2_images_before_averaging_grid)
axarr[4].set_title('2nd query images before avg', fontsize=50)
axarr[4].axis('off')

#### PLOT Q2 IMAGES AFTER AVERAGING ####
q_2_images_after_averaging = torch.load(os.path.join(args.log_dir, "q_2_images_after_averaging.pt"))
q_2_images_after_averaging_grid = convert_image_np(torchvision.utils.make_grid(q_2_images_after_averaging[indices], nrow=len(indices)))     
axarr[5].imshow(q_2_images_after_averaging_grid)
axarr[5].set_title('2nd query images after avg', fontsize=50)
axarr[5].axis('off')

if args.total_sigma is not None:
    total_sigma = "total_sigma_"+str(args.total_sigma)
    f.suptitle(total_sigma, fontsize=45)
plt.tight_layout()
f.savefig(os.path.join(args.log_dir, "pipeline.png"))
plt.close(f)
