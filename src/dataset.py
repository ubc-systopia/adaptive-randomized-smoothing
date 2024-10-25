from torchvision import transforms, datasets
from torchvision.datasets import CelebA
from typing import *
import torch
import os
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# list of all datasets
DATASETS = ["cifar10",  "celeba", "imagenet"]

def get_dataset(dataset: str, split: str, path: str, transform_params: dict = {}) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset.lower() == "cifar10":
        return _cifar10(split, path, transform_params)
    elif dataset.lower() == "celeba":
        return _celeba(split, path, transform_params)
    elif dataset.lower() == "imagenet":
        return _imagenet(split, path)

# CIFAR-10 (32x32x3 images)
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

#  BG20k MEAN and STD DEV for 48x48x3 images (i.e. k = 48)
BG20K_48_48_MEAN = [0.43384078, 0.43419835, 0.4105737]
BG20K_48_48_STDDEV = [0.20452765, 0.19125941, 0.20084971]

#  BG20k MEAN and STD DEV for 64x64x3 images (i.e. k = 64)
BG20K_64_64_MEAN = [0.43382293, 0.43418652, 0.4105753]
BG20K_64_64_STDDEV = [0.20860474, 0.1945744, 0.20410348]

#  BG20k MEAN and STD DEV for 96x96x3 images (i.e. k = 96)
BG20K_96_96_MEAN = [0.43382654, 0.4341659 , 0.41056666]
BG20K_96_96_STDDEV = [0.21364133, 0.19869995, 0.20809236]

# CELEBA
_CELEBA_MEAN = [0.5063, 0.4258, 0.3832]
_CELEBA_STDDEV = [0.3098, 0.2897, 0.2890]

# IMAGENET
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

# BG20k MEAN and STD DEV below are calculated by 
# running scripts/preprocess_bg20k/compute_mean_std.py
# you can use other values of image_size, but be sure to calculate the mean and std using this .py file

def get_normalize_layer(dataset: str, image_size: int) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset.lower() == "cifar10":
        cifar10_image_fraction = (32*32) / (image_size*image_size) # this is applied per channel
        bg20k_image_fraction = 1 - cifar10_image_fraction
        if image_size == 32:
            # no extra background padding or superimposition business; just return plain old CIFAR-10 as is
            SUPERIMPOSED_MEAN, SUPERIMPOSED_STDDEV = _CIFAR10_MEAN, _CIFAR10_STDDEV
        elif image_size == 48:
            SUPERIMPOSED_MEAN = (np.multiply(cifar10_image_fraction, _CIFAR10_MEAN) + np.multiply(bg20k_image_fraction, BG20K_48_48_MEAN)).tolist()
            SUPERIMPOSED_STDDEV = np.sqrt((cifar10_image_fraction * np.square(_CIFAR10_STDDEV)) + (bg20k_image_fraction * np.square(BG20K_48_48_STDDEV))).tolist()
        elif image_size == 64:
            SUPERIMPOSED_MEAN = (np.multiply(cifar10_image_fraction, _CIFAR10_MEAN) + np.multiply(bg20k_image_fraction, BG20K_64_64_MEAN)).tolist()
            SUPERIMPOSED_STDDEV = np.sqrt((cifar10_image_fraction * np.square(_CIFAR10_STDDEV)) + (bg20k_image_fraction * np.square(BG20K_64_64_STDDEV))).tolist()
        elif image_size == 96:
            SUPERIMPOSED_MEAN = (np.multiply(cifar10_image_fraction, _CIFAR10_MEAN) + np.multiply(bg20k_image_fraction, BG20K_96_96_MEAN)).tolist()
            SUPERIMPOSED_STDDEV = np.sqrt((cifar10_image_fraction * np.square(_CIFAR10_STDDEV)) + (bg20k_image_fraction * np.square(BG20K_96_96_STDDEV))).tolist()
        else:
            raise ValueError('for image_size not in (32,48,64,96), you will need to compute the mean and std, then add an extra elif condition')
        return NormalizeLayer(SUPERIMPOSED_MEAN, SUPERIMPOSED_STDDEV)
    elif dataset.lower() == "celeba":
        return NormalizeLayer(_CELEBA_MEAN, _CELEBA_STDDEV)
    elif dataset.lower() == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)


###########################cifar10############################
    
class PadAndShift(object):
    def __init__(self, transform_params: dict = {}, split: str = "train"):
        # add parameters of the location of the image
        self.pad_size = transform_params["pad_size"] # one sided pad_size; hence total padding = 2 * pad_size
        self.num_image_locations = transform_params["num_image_locations"] # options = ["edges", "random"]
        self.background = transform_params["background"] # options = ["black", "nature"]
        self.mask_supervise = transform_params["mask_supervise"]
        self.padded_img_path = transform_params["padded_img_path"]

        # ideally should be able to adjust the dimensions of background image as welll but right now they are fixed to 48*48
        if self.background == "nature":
            print("using nature background!")
            if split == "train":
                # path to bg-20k train dataset images
                self.random_bg_image_paths = glob.glob(os.path.join(self.padded_img_path, "train", "*"))
            elif split == "test":
                # path to bg-20k test dataset images
                self.random_bg_image_paths = glob.glob(os.path.join(self.padded_img_path, "test", "*"))
                
    def __call__(self, image):
        # assume the image is of PIL form, hence first convert it to a numpy array
        image = np.array(image)
        h, w, _ = image.shape
        new_h = h + 2 * self.pad_size
        new_w = w + 2 * self.pad_size

        if self.pad_size == 0: #no padding
           padded_image = image
           x_prime, y_prime = 0, 0 
        else:
            # generate the padded image based on chosen background
            if self.background == "black" or self.background == None:
                padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            elif self.background == "nature":
                random_bg_image_path = random.choice(self.random_bg_image_paths)     
                bg_img = Image.open(random_bg_image_path)
                bg_img = bg_img.resize((new_w, new_h))
                padded_image = np.array(bg_img)

            # if shifting, choose random locations or static location if not
            if self.num_image_locations == "edges":
                a = random.choice([0, 2*self.pad_size])
                b = random.choice(np.arange(0, 2*self.pad_size))
                flip = np.random.binomial(1,0.5)
                if flip == 0:
                    x_prime, y_prime = a, b
                else:
                    x_prime, y_prime = b, a
            elif self.num_image_locations == "random":
                # place image randomly anywhere within the padded image
                x_prime, y_prime = np.random.randint(low=0, high=2*self.pad_size, size=2)
            else:
                raise Exception("Choose 'edges' or 'random'")
        
            # put the image as per shifting choice
            if self.background in ["black", "nature", None]:
                padded_image[x_prime:x_prime + h, y_prime:y_prime + w, :] = image
        
        if self.mask_supervise > 0:
            ground_truth_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            ground_truth_mask[x_prime:x_prime + h, y_prime:y_prime + w] = 1
            return padded_image, ground_truth_mask

        return padded_image
    
# a fix for cifar to enable mask supervision (return data and ground truth mask)
class mycifar(datasets.CIFAR10):
    def __getitem__(self, index: int) -> torch.Tuple[torch.Any, torch.Any, torch.Any]:
        image, target =  super().__getitem__(index)
        padded_image, ground_truth_mask = transforms.functional.to_tensor(image[0]), torch.from_numpy(image[1]).float()  #np.ndarray to tensor
        # using transforms.functional.to_tensor gives torch.Size([3, 48, 48]); using torch.from_numpy gives torch.Size([48, 48, 3])
        # to match transforms.ToTensor(), should use transforms.functional.to_tensor

        return padded_image, target, ground_truth_mask

def _cifar10(split: str, path: str, transform_params: dict) -> Dataset:

    if split.lower() == "train":
        train_bool = True
        if transform_params["mask_supervise"] > 0:
            CIFAR_TRAIN_TRANSFORM = transforms.Compose([
                                    PadAndShift(transform_params, split=split), # the split param is only required for bg-20k nature bg dataset
                                ])
            dataset = mycifar(path, train=train_bool, download=True, transform=CIFAR_TRAIN_TRANSFORM)
        else : 
            CIFAR_TRAIN_TRANSFORM = transforms.Compose([
                                    PadAndShift(transform_params, split=split), # the split param is only required for bg-20k nature bg dataset
                                    transforms.ToTensor()
                                ])
            dataset = datasets.CIFAR10(path, train=train_bool, download=True, transform=CIFAR_TRAIN_TRANSFORM)
    elif split.lower() == "test":
        train_bool = False
        
        if transform_params["mask_supervise"] > 0:
            CIFAR_TEST_TRANSFORM = transforms.Compose([
                                    PadAndShift(transform_params, split=split), # the split param is only required for bg-20k nature bg dataset
                                ])
            dataset = mycifar(path, train=train_bool, download=True, transform=CIFAR_TEST_TRANSFORM)
        else : 
            CIFAR_TEST_TRANSFORM = transforms.Compose([
                                    PadAndShift(transform_params, split=split), # the split param is only required for bg-20k nature bg dataset
                                    transforms.ToTensor()
                                ])
            dataset = datasets.CIFAR10(path, train=train_bool, download=True, transform=CIFAR_TEST_TRANSFORM)

    return dataset


###########################celeba############################

class CropAroundMouth(object):
    def __init__(self, transform_params: dict = {}):
        # add parameters of the location of the image
        self.pad_size = transform_params["pad_size"] # area around mouth
        self.num_image_locations = transform_params["num_image_locations"]
        self.face_feature = transform_params["face_feature"]

    def __call__(self, tuple):
        new_h, new_w = 160, 160
        image, landmarks, index = tuple[0], tuple[1], tuple[2]
        w, h = image.size
        # upscaling/downscaling
        scale = 1.5*((new_w / w) if w < h else (new_h / h))
        image = transforms.Resize(size=(int(h*scale),int(w*scale))).forward(image)
        # assume the image is of PIL form, hence first convert it to a numpy array
        image = np.array(image)
        h, w, _ = image.shape
        landmarks = np.array(landmarks, dtype=float)
        if self.face_feature == "eyes":
            eyes = (scale*landmarks[:4]).astype(int) # left_x, left_y, right_x, right_y
            # get eyes boundary
            boundaries = np.array([[max(0, min(h-new_h, max(eyes[3],eyes[1]) + 2*self.pad_size - new_h)),
                                    min(h-new_h, max(0, min(eyes[3],eyes[1]) - 2*self.pad_size))],
                                   [max(0, min(w-new_w, eyes[2] + self.pad_size - new_w)),
                                    min(w-new_w, max(0, eyes[0] - self.pad_size))]])
        else:
            mouth = (scale*landmarks[6:]).astype(int) # left_x, left_y, right_x, right_y
            # get mouth boundary
            boundaries = np.array([[max(0, min(h-new_h, max(mouth[3],mouth[1]) + 2*self.pad_size - new_h)),
                                    min(h-new_h, max(0, min(mouth[3],mouth[1]) - 2*self.pad_size))],
                                   [max(0, min(w-new_w, mouth[2] + self.pad_size - new_w)),
                                    min(w-new_w, max(0, mouth[0] - self.pad_size))]])

        if boundaries[0,0] > boundaries[0,1] or boundaries[1,0] > boundaries[1,1]:
            print(f"ERROR: {boundaries[0,0]} > {boundaries[0,1]}, or {boundaries[1,0]} > {boundaries[1,1]}, mouth has position {mouth} at index {index}")
            # if the data has a mis-label, then we might get messed up boundaries
            if boundaries[0,0] > boundaries[0,1]:
                boundaries[0,1] = boundaries[0,0]
            if boundaries[1,0] > boundaries[1,1]:
                boundaries[1,1] = boundaries[1,0]
 
        if h < new_h or w < new_w:
            print(w, h, new_w, new_h)
            raise RuntimeError(f"MAGIC NUMBER: {new_w, new_h} does not suffice")

        # if shifting, choose random transformations
        if self.pad_size > 0:
            if self.num_image_locations == "random":
                # place image randomly anywhere within the padded image
                if boundaries[0,0] == boundaries[0,1]:
                    x_prime = boundaries[0,0]
                else:
                    x_prime = np.random.randint(low=boundaries[0,0], high=boundaries[0,1])
                if boundaries[1,0] == boundaries[1,1]:
                    y_prime = boundaries[1,0]
                else:
                    y_prime = np.random.randint(low=boundaries[1,0], high=boundaries[1,1])
            else:
                raise Exception("celeba experiments only support random location ")
        else:
            x_prime, y_prime = 0, 0
        # This is for checking that the boundary is correct
        padded_image = image[x_prime:(x_prime+new_w), y_prime:(y_prime+new_h)]
        return padded_image

# A fix for CelebA
class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, "img_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f'Target type "{t}" is not recognized.')
        if self.transform is not None: # Added this thing required to crop face to mouth
            out = self.transform((X, self.landmarks_align[index, :], index))
            X = transforms.functional.to_tensor(out)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def _check_integrity(self) -> bool:
        return True

def _celeba(split: str, path: str, transform_params: dict) -> Dataset:
    """
    """
    # Assume already downloaded (manually)
    CELEBA_TRANSFORM = transforms.Compose([
        CropAroundMouth(transform_params)
    ])
    dataset = MyCelebA(
        root=path,
        split=split.lower(),
        download=False,
        transform=CELEBA_TRANSFORM,
    )
    return dataset


###########################imagenet############################

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your test directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/testprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
def _imagenet(split: str, path: str) -> Dataset:
    if IMAGENET_LOC_ENV in os.environ:
        dir = os.environ[IMAGENET_LOC_ENV]
    elif path is not None:
        dir = path
    else:
        raise ValueError("environment variable for ImageNet directory not set and data path in yaml is None")

    if split == "train":
        subdir = os.path.join(dir, "ILSVRC2012_train/data")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "ILSVRC2012_validation/data")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)
    # return TinyDataset(datasets.ImageFolder(subdir, transform))

class TinyDataset(Dataset):
  def __init__(self, dset, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dset = dset
  def __len__(self):
    return 1000
  def __getitem__(self, index):
    return self.dset[index]


########################### dataset utils ############################

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

class PreProcessLayer(torch.nn.Module):
    """Transforms the input with the following transformations
    1) RandomHorizontalFlip
    """
   
    def __init__(self, prob_flip=0.5):
        """
        :param prob_flip: prob with which it is flipped
        """
        super(PreProcessLayer, self).__init__()
        self.pre_transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=prob_flip)
                        ])

    def forward(self, input: torch.tensor):
        input = self.pre_transforms(input)
        return input
