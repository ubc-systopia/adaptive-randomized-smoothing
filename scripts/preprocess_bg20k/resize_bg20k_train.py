import os
import glob
from PIL import Image
import argparse

"""
sample command for different k:
python scripts/preprocess_bg20k/resize_bg20k_train.py -k 40
python scripts/preprocess_bg20k/resize_bg20k_train.py -k 48
python scripts/preprocess_bg20k/resize_bg20k_train.py -k 64
python scripts/preprocess_bg20k/resize_bg20k_train.py -k 96
python scripts/preprocess_bg20k/resize_bg20k_train.py -k 128
python scripts/preprocess_bg20k/resize_bg20k_train.py -k 256
python scripts/preprocess_bg20k/resize_bg20k_train.py -k 512
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('-k', type=int, help='new size of the downsampled images')
args = parser.parse_args()

bg_20k_dir = os.path.join(os.getenv('SCRATCH_DIR'), "bg_20k")
resized_dir_name = str(args.k)+"_"+str(args.k)+"_size"

# resize train images
train_og_res_images = os.path.join(bg_20k_dir, "original_size/train/*")
train_og_res_image_paths = glob.glob(os.path.join(train_og_res_images))

train_new_res_dir = os.path.join(bg_20k_dir, resized_dir_name, "train")
if not os.path.exists(train_new_res_dir):
    os.makedirs(train_new_res_dir)

for og_res_image_path in train_og_res_image_paths:
    og_res_image_filename = og_res_image_path.split("/")[-1].split(".")[0]
    og_res_image = Image.open(og_res_image_path)
    
    new_res_image_filename = og_res_image_filename+"_"+str(args.k)+"_"+str(args.k)+".jpg"
    new_res_image = og_res_image.resize((args.k, args.k))
    new_res_image.save(os.path.join(train_new_res_dir, new_res_image_filename))
    print("done with train "+og_res_image_filename)