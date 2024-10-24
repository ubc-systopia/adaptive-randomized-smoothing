import os
import math
import torch
import random
import numpy as np
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import datetime
import matplotlib.pyplot as plt
import torchvision

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# def learning_rate(init, epoch):
#     optim_factor = 0
#     if(epoch > 160):
#         optim_factor = 3
#     elif(epoch > 120):
#         optim_factor = 2
#     elif(epoch > 60):
#         optim_factor = 1
#     return init*math.pow(0.2, optim_factor)

# def make_directories(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

def get_image_size(conf):
    if conf.data.dataset == "cifar10":
        image_size = 32
    elif conf.data.dataset == "celeba":
        return 160
    elif conf.data.dataset == "imagenet":
        image_size = 224
    if conf.data.transform.pad_size > 0:
        image_size += (2 * conf.data.transform.pad_size)
    return image_size

def get_save_directory_path(args, conf, now):
    # get save directory to store logs
    # seed = "_seed_"+str(args.seed)
    # cfg_fname = os.path.split(args.yaml[0])[-1]
    # cfg_name = os.path.splitext(cfg_fname)[0]
    run_description = str(conf.run_description)
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # save_dir = os.path.join("logs", now + "_"+ cfg_name + "_" + run_description)
    save_dir = os.path.join("logs", now + "_" + run_description)
    return save_dir

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

def get_optimizer(params, optim_dict):
    if optim_dict["optimizer"] == "SGD":
        print('optimizer is : SGD') 
        return SGD(params,
                   lr=optim_dict["lr"],
                   momentum=optim_dict["momentum"],
                   weight_decay=optim_dict["weight_decay"])
    elif optim_dict["optimizer"] == "Adam":
        print('optimizer is : Adam') 
        return Adam(params,
                    lr=optim_dict["lr"],
                    # eps=optim_dict["eps"],
                    weight_decay=optim_dict["weight_decay"])
    elif optim_dict["optimizer"] == "AdamW":
        print('optimizer is : AdamW') 
        return AdamW(params,
                    lr=optim_dict["lr"],
                    weight_decay=optim_dict["weight_decay"])

def get_scheduler(optimizer, scheduler_dict, epoch):
    if scheduler_dict["scheduler"] == "StepLR":
        print('scheduler is : StepLR')
        return StepLR(optimizer, step_size=scheduler_dict["step_size"], gamma=scheduler_dict["gamma"])
    elif scheduler_dict["scheduler"] == "CosineAnnealingLR":
        print('scheduler is : CosineAnnealingLR')
        return CosineAnnealingLR(optimizer, T_max=epoch)

# process images
def convert_image_np(tens):
    return tens.data.cpu().numpy().transpose((1, 2, 0))

# def clip_image(img):
#     return (img * 255).astype(np.uint8)

# def vizualize_images(conf, saved_dir):  #TODO
#     with torch.no_grad():
        
#         if conf.arch.num_query == 1 and conf.arch.mask_type == "vanilla":
#             num_cols = 2
#         elif conf.arch.num_query == 1 and conf.arch.mask_type == "static":
#             num_cols = 3
#         elif conf.arch.num_query == 2:
#                 num_cols = 5
            
#         # Plot the results side-by-side
#         f, axarr = plt.subplots(1, num_cols)
#         col_index = 0

#         # original images
#         original_images = torch.load(
#             os.path.join(saved_dir, "original_images.pt"))
#         original_images_grid = convert_image_np(
#             torchvision.utils.make_grid(original_images[:100]))    
#         axarr[col_index].imshow(clip_image(original_images_grid))
#         axarr[col_index].set_title('Original')
#         col_index += 1

#         # post first query images
#         images_after_first_query = torch.load(
#             os.path.join(saved_dir, "q_1_images.pt"))
#         images_after_first_query_grid = convert_image_np(
#             torchvision.utils.make_grid(images_after_first_query[:100]))
#         axarr[col_index].imshow(clip_image(images_after_first_query_grid))
#         axarr[col_index].set_title('1q')
#         col_index += 1
        
#         if conf.arch.num_query == 2:
#             second_query_masks = torch.load(os.path.join(
#                 saved_dir, "post_clamped_second_query_masks.pt"))
#             second_query_masks_grid = convert_image_np(
#                 torchvision.utils.make_grid(second_query_masks[:100]))            
#             axarr[col_index].imshow(clip_image(second_query_masks_grid))
#             axarr[col_index].set_title('2q masks')
#             col_index += 1
        
#             if conf.arch.average_queries:
#                 q_2_images_before_averaging = torch.load(
#                     os.path.join(saved_dir, "q_2_images_before_averaging.pt"))
#                 q_2_images_before_averaging_grid = convert_image_np(
#                     torchvision.utils.make_grid(q_2_images_before_averaging[:100]))
#                 axarr[col_index].imshow(clip_image(q_2_images_before_averaging_grid))
#                 axarr[col_index].set_title('2q bef avg')
#                 col_index += 1
                
#                 q_2_images_after_averaging = torch.load(
#                     os.path.join(saved_dir, "q_2_images_after_averaging.pt"))
#                 q_2_images_after_averaging_grid = convert_image_np(
#                     torchvision.utils.make_grid(q_2_images_after_averaging[:100]))
#                 axarr[col_index].imshow(clip_image(q_2_images_after_averaging_grid))
#                 axarr[col_index].set_title('2q aft avg')
#                 col_index += 1
#             else:
#                 q_2_images = torch.load(os.path.join(saved_dir, "q_2_images.pt"))
#                 q_2_images_grid = convert_image_np(
#                     torchvision.utils.make_grid(q_2_images[:100]))
#                 axarr[col_index].imshow(clip_image(q_2_images_grid))
#                 axarr[col_index].set_title('2q')
#                 col_index += 1

#         if conf.arch.mask_type == "static":
#             static_mask_tensor = torch.load(
#                 os.path.join(saved_dir, "static_mask.pt"))
#             static_mask_np = static_mask_tensor.data.cpu().numpy()
#             img = axarr[col_index].pcolormesh(static_mask_np, vmin=0., vmax=1.)
#             axarr[col_index].invert_yaxis()
#             axarr[col_index].set_aspect("equal")
#             axarr[col_index].set_title('mask (1st)')
#             f.colorbar(img, ax=axarr[col_index], label="weight", orientation="horizontal", ticks=[
#                 0., 0.25, 0.5, 0.75, 1.])

#         # plt.title(saved_dir.split()[-1])
#         plt.tight_layout()
#         f.savefig(os.path.join(saved_dir, "first_batch.png"))
#         plt.close(f)
