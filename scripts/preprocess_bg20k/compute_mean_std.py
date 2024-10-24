import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse

"""
sample command:
python scripts/preprocess_bg20k/compute_mean_std.py -images_dir '/bg_20k/train/original_size/'
python scripts/preprocess_bg20k/compute_mean_std.py -images_dir '/bg_20k/train/48_48_size/'
python scripts/preprocess_bg20k/compute_mean_std.py -images_dir '/bg_20k/train/64_64_size/'
python scripts/preprocess_bg20k/compute_mean_std.py -images_dir '/bg_20k/train/96_96_size/'
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('-images_dir', type=str, help='path of directory where all images are stored')
args = parser.parse_args()

class CustomImageDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
# Define the dataset path and transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the custom dataset
dataset = CustomImageDataset(images_dir=args.images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=4)

# Initialize the sum and squared sum and number of batches
mean = 0.0
std = 0.0
nb_samples = 0

print("starting to compute...")
for data in dataloader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')

np.save(os.path.join(args.images_dir, "mean.npy"), mean)
np.save(os.path.join(args.images_dir, "std.npy"), std)