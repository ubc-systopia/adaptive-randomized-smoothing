import os
import torch 
import torchvision 
import matplotlib.pyplot as plt
import numpy as np
import argparse

"""
sample commands:
python scripts/plot/pipeline_v2.py --k 64 --total_sigma 0.12 --indices deterministic --log_dir ""
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
parser.add_argument('--k', type=int, default="None", help='height/width of the image')
parser.add_argument('--total_sigma', type=float, default="None", help='total sigma value for the plots')
parser.add_argument('--indices', type=str, help='random or deterministic')
parser.add_argument('--log_dir', type=str, help='root path to pick up all pt files from and also store the plot')
args = parser.parse_args()

# choose 5 random indices
if args.indices == "random":
    indices = np.random.choice(100, size=5) # 100 should be smaller than num_to_store in the yaml file you used
elif args.indices == "deterministic": #you can make your own choice
    if args.k == 32 or args.k == 48:
        indices = np.array([10, 20, 30, 40, 50])
    elif args.k == 64:
        indices = np.array([15, 30, 45, 60, 75])
    elif args.k == 96:
        indices = np.array([20, 40, 60, 80, 99])

original_images = torch.load(os.path.join(args.log_dir, "original_images.pt"), map_location=torch.device('cpu'))[indices].unsqueeze(dim=0)
q_1_noisy_images = torch.load(os.path.join(args.log_dir, "q_1_images.pt"), map_location=torch.device('cpu'))[indices].unsqueeze(dim=0)
second_query_masks = torch.load(os.path.join(args.log_dir, "post_clamped_second_query_masks.pt"), map_location=torch.device('cpu'))[indices]
# normalizing the masks to emphasize if the max mask weight is tiny compared to 1.0 
for i in range(len(second_query_masks)):
    second_query_masks[i] = second_query_masks[i] / torch.max(second_query_masks[i])
second_query_masks_reshaped = second_query_masks.unsqueeze(dim=0).repeat((1, 1, 3, 1, 1))
second_query_masks = second_query_masks.repeat((1, 3, 1, 1))
masked_images = torch.mul(original_images, second_query_masks_reshaped)
q_2_images_before_averaging = torch.load(os.path.join(args.log_dir, "q_2_images_before_averaging.pt"), map_location=torch.device('cpu'))[indices].unsqueeze(dim=0)
q_2_images_after_averaging = torch.load(os.path.join(args.log_dir, "q_2_images_after_averaging.pt"), map_location=torch.device('cpu'))[indices].unsqueeze(dim=0)

stacked = torch.vstack((original_images,
                        q_1_noisy_images,
                        second_query_masks_reshaped,
                        masked_images,
                        q_2_images_before_averaging,
                        q_2_images_after_averaging))
stacked_reshaped = stacked.view(-1, 3, args.k, args.k)

# # Make a grid of n x 5 images (where n is whatever)
grid = torchvision.utils.make_grid(stacked_reshaped, nrow=5, padding=2, pad_value=1)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(grid.permute(1, 2, 0))

if args.total_sigma is not None:
    total_sigma = "total_sigma_"+str(args.total_sigma)
    # f.suptitle(total_sigma, fontsize=45)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, "pipeline_"+total_sigma+".png"), bbox_inches='tight')
plt.close()
