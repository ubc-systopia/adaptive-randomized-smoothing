from __future__ import print_function
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import sys
sys.path.append("..")
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.nn.functional import sigmoid
from datetime import datetime
from torchsummary import summary
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!

from utils import (set_seed, get_image_size, get_save_directory_path,
                   init_logfile, log, get_optimizer, get_scheduler)
from dataset import get_dataset
from models.two_query_arch import TWO_QUERY_ARCH
from models.single_query_arch import SINGLE_QUERY_ARCH


######################################################################


def train(epoch, conf, saved_dir, train_loader, model):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if conf.trainer.mask_supervise > 0:
            data, targets, ground_truth_mask = batch[0], batch[1], batch[2]
            data, targets, ground_truth_mask = data.to(conf.trainer.device), targets.to(conf.trainer.device), ground_truth_mask.to(conf.trainer.device)
        else:
            data, targets = batch[0], batch[1]
            data, targets = data.to(conf.trainer.device), targets.to(conf.trainer.device)

        if conf.data.dataset == 'celeba' and conf.data.multilabel_overwrite is not None:
            targets = targets[:, conf.data.multilabel_overwrite].view(-1, 1).float()

        logging_trackers = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'mode': 'train',
            'saved_dir': saved_dir,
            # 'sigma_log_file': train_sigma_log_file,
        }

        # reconstruction loss for mask to learn
        if conf.trainer.mask_recon > 0 :
            output_pred = model(data, logging_trackers)
            mask_outputs_new = model.mask_model(data)
            mask_out = mask_outputs_new[:,0:1,:,:]
            recon = mask_outputs_new[:,1:4,:,:]
        elif conf.trainer.mask_supervise > 0:
            output_pred, mask = model(data, logging_trackers)
        else:
            output_pred = model(data, logging_trackers)
            
        # compute loss
        if args.resume_mask_only is None: 
            classifier_optimizer.zero_grad()
        if len(mask_params) > 0:
            mask_optimizer.zero_grad()
        total_loss = criterion(output_pred, targets)

        # add reconstruction loss for original image
        if conf.trainer.mask_recon > 0:
            total_loss += conf.trainer.mask_recon * (MSELoss().to(conf.trainer.device))(recon, data)
        
        # add ground truth mask supervision
        if conf.trainer.mask_supervise > 0:
            temp = (MSELoss().to(conf.trainer.device))(mask, ground_truth_mask)
            total_loss += (conf.trainer.mask_supervise *  temp)

        log(loss_file, "{}".format(total_loss))
            
        total_loss.backward()

        # optim step
        if args.resume_mask_only is None:
            classifier_optimizer.step()
        if len(mask_params) > 0:
            mask_optimizer.step()

        # clamp mask and budget frac values between 0 and 1 (PGD step)
        with torch.no_grad():
            if conf.arch.mask_type == "static":
                model.static_mask.clamp_(min=0., max=1.)

        # if batch_idx % 30 == 0:
        #     if conf.data.dataset == "celeba":
        #         pred = (sigmoid(output_pred) > 0.5).int()
        #     else:
        #         pred = output_pred.max(1, keepdim=True)[1]
        #     correct = pred.eq(targets.view_as(pred)).sum().item() / len(data)
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f} Acc: {:.2f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), total_loss.item(), correct))


def test(epoch, conf, saved_dir, test_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        for batch_idx, batch in enumerate(test_loader):
            if conf.trainer.mask_supervise > 0:
                data, targets, ground_truth_mask = batch[0], batch[1], batch[2]
                data, targets, ground_truth_mask = data.to(conf.trainer.device), targets.to(conf.trainer.device), ground_truth_mask.to(conf.trainer.device)
            else:
                data, targets = batch[0], batch[1]
                data, targets = data.to(conf.trainer.device), targets.to(conf.trainer.device)

            if conf.data.dataset == 'celeba' and conf.data.multilabel_overwrite is not None:
                targets = targets[:, conf.data.multilabel_overwrite].view(-1, 1).float()

            logging_trackers = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'mode': 'test',
                'saved_dir': saved_dir,
                # 'sigma_log_file': test_sigma_log_file,
            }
            
            # create n copies of each example
            n = conf.trainer.test_sample_copy
            repeated_batch = data[:, None, :, :, :].repeat((1, n, 1, 1, 1)).flatten(0,1)
            if conf.trainer.mask_supervise > 0:
                outputs, _ = model(repeated_batch, logging_trackers)
            else : 
                outputs = model(repeated_batch, logging_trackers)
            
            if conf.data.dataset == "celeba":
                all_predictions = (sigmoid(outputs) > 0.5).int()
            else:
                all_predictions = outputs.argmax(axis=1)
            all_predictions = all_predictions.reshape(conf.trainer.test_batch_size, n).cpu().numpy()

            minlength = 2 if conf.data.num_classes == 1 else conf.data.num_classes
            counts = np.apply_along_axis(np.bincount, axis=1, arr=all_predictions, minlength=minlength)
            predictions = np.argmax(counts, axis=1)
            correct += np.equal(predictions, targets.cpu().numpy().flatten()).sum()
        
        # save a bunch of metrics
        test_acc = 100. * (correct / len(test_loader.dataset))
        log(logfilename, "{} \t {:.3}".format(epoch, test_acc))
        # print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), test_acc))
    return test_acc

######################################################################


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument("-s", "--seed", type=int, default=42, help="random seed")
    argparser.add_argument("-y", "--yaml", default=list(), help="paths to base configs. Loaded from left-to-right.")
    argparser.add_argument("-r", "--resume_mask_only", default=None, help="for imagenet ars training, to load the pretrained model and train the adaptive mask model only, should be a path to model ckpt, e.g. logs/cifar_sig05_cohen/model_sd.pt")

    args = argparser.parse_args()
    conf = OmegaConf.load(args.yaml)
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    print(args)
    # set the seeds
    set_seed(args.seed)
    save_dir = get_save_directory_path(args, conf, now) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # add args and save config file
    conf.seed = args.seed   #add command line seed to the conf; the reason I did this is bc I don't have to generate different yamls for different seed, but can still save the seed value in the conf
    conf.resume_mask_only = args.resume_mask_only
    OmegaConf.save(conf, os.path.join(save_dir, "{}-config.yaml".format(now)))

    # DATASET TRANSFORMATION
    transform_params = {
        "pad_size": conf.data.transform.pad_size,
        "padded_img_path": conf.data.transform.padded_img_path,
        "num_image_locations": conf.data.transform.num_image_locations,
        "background": conf.data.transform.background,
        "mask_supervise": conf.trainer.mask_supervise,
        "face_feature": conf.data.transform.face_feature
    }

    # get training/test datasets and loaders
    train_dataset = get_dataset(dataset=conf.data.dataset,
                                split="train",
                                path=os.path.join(conf.data.dataset_path, conf.data.dataset),
                                transform_params=transform_params)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=conf.trainer.train_batch_size, num_workers=4, drop_last=True)
    test_dataset = get_dataset(dataset=conf.data.dataset,
                               split='test',
                               path=os.path.join(conf.data.dataset_path, conf.data.dataset),
                               transform_params=transform_params)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=conf.trainer.test_batch_size, num_workers=4, drop_last=True)
    

    if conf.arch.num_query == 1:
        model = SINGLE_QUERY_ARCH(conf).to(conf.trainer.device)
    elif conf.arch.num_query == 2:
        model = TWO_QUERY_ARCH(conf, args).to(conf.trainer.device) 
    else :
        raise Exception('num_query > 2 not implemented')

    image_size_assumed = get_image_size(conf) 
    print("assumed image size is : ", image_size_assumed)
    image_size = train_dataset[0][0].shape[-1]
    print("real image size is : ", image_size)
    assert image_size == image_size_assumed

    # if conf.arch.num_query == 2:
    #     mask_model_summary = summary(model.mask_model, (3, image_size, image_size)) 
        # torch.save(mask_model_summary, os.path.join(save_dir, "mask_model_summary.pt"))    
    
    # base_classifier_summary = summary(model.base_classifier, (3, image_size, image_size))
    # torch.save(base_classifier_summary, os.path.join(save_dir, "base_classifier_summary.pt"))
        
    # print(model)

    ######################################################################
    # Training the model:
    ######################################################################

    # all transformation parameters
    mask_params = []
    mask_conf = dict(conf.trainer.mask) # convert <class 'omegaconf.dictconfig.DictConfig'> to <class 'dict'>
    if conf.arch.num_query == 1 and conf.arch.mask_type == "vanilla":
        print('doing vanilla! not training mask.')
        pass
    elif conf.arch.num_query == 1 and conf.arch.mask_type == "static":
        mask_params.append({'params': model.static_mask})  
        print('doing static! 1 query and learn an average mask in the query')
        mask_optimizer = get_optimizer(mask_params, mask_conf)
        mask_scheduler = get_scheduler(mask_optimizer, mask_conf, conf.trainer.epoch)  
    elif conf.arch.num_query == 2 and conf.arch.mask_type == "adaptive":
        mask_params.append({'params': model.mask_model.parameters()})
        print('doing ARS!')
        mask_optimizer = get_optimizer(mask_params, mask_conf)
        mask_scheduler = get_scheduler(mask_optimizer, mask_conf, conf.trainer.epoch) 
    else:
        raise ValueError('num_query or mask_type might be wrong in the yaml file!')

    # base classifier parameters
    if args.resume_mask_only is None:
        classifier_params = [{'params': model.base_classifier.parameters()}]
        print('train the base classifier!')
        classifier_conf = dict(conf.trainer.classifier)
        classifier_optimizer = get_optimizer(classifier_params, classifier_conf)
        classifier_scheduler = get_scheduler(classifier_optimizer, classifier_conf, conf.trainer.epoch) 
    
    # create loss function
    criterion = CrossEntropyLoss().to(conf.trainer.device) if conf.data.dataset.lower() != 'celeba' else BCEWithLogitsLoss().to(conf.trainer.device)

    # create train log file
    logfilename = os.path.join(save_dir, 'train_log.txt')
    init_logfile(logfilename, "epoch \t test_acc")

    # option to save sigmas in log
    # train_sigma_log_file = os.path.join(save_dir, 'train_sigma_log.txt')
    # test_sigma_log_file = os.path.join(save_dir, 'test_sigma_log.txt')
    # if conf.arch.num_query == 1:
    #     init_logfile(train_sigma_log_file, 'sigma_1')
    #     init_logfile(test_sigma_log_file, 'sigma_1')
    # elif conf.arch.num_query == 2:
    #     init_logfile(train_sigma_log_file, 'sigma_1 \t sigma_2 \t total_sigma')
    #     init_logfile(test_sigma_log_file, 'sigma_1 \t sigma_2 \t total_sigma')
    #     budget_query_split_log_file = os.path.join(save_dir, 'budget_query_split_log.txt')
    #     init_logfile(budget_query_split_log_file, 'query_1 \t query_2')

    loss_file = os.path.join(save_dir, 'train_loss.txt')
    init_logfile(loss_file, 'total_loss')

    best_test_acc = 0.0
    for epoch in tqdm(range(1, conf.trainer.epoch + 1)):
        # all to store less data
        if epoch % 10 == 0 or epoch == 1:
            train_saved_dir = os.path.join(save_dir, "train", "epoch_"+str(epoch))
            test_saved_dir = os.path.join(save_dir, "test", "epoch_"+str(epoch))
            if not os.path.exists(train_saved_dir):
                os.makedirs(train_saved_dir)
            if not os.path.exists(test_saved_dir):
                os.makedirs(test_saved_dir)

        # print("Time when training started {}", datetime.now().strftime("%H:%M:%S"))
        train(epoch, conf, train_saved_dir, train_loader, model)
        print("done training epoch {}!".format(epoch))
        # print("Time when training ended {}", datetime.now().strftime("%H:%M:%S"))
        curr_test_acc = test(epoch, conf, test_saved_dir, test_loader, model)
        print("done evaluating epoch {}!".format(epoch))
        # print("Time when testing ended {}", datetime.now().strftime("%H:%M:%S"))
        
        # save the best model
        if curr_test_acc > best_test_acc:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_sd.pt"))
        
        if args.resume_mask_only is None:
            classifier_scheduler.step()
        if len(mask_params) > 0:
            mask_scheduler.step()
