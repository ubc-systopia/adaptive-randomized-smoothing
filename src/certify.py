# evaluate a smoothed classifier on a dataset
import torch
import datetime
import os
import math
import time
import torch
import numpy as np
import torch.nn as nn
import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.nn.functional import sigmoid
import torch.distributions as D
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import sys
sys.path.append("..")

from dataset import get_dataset
from utils import set_seed, get_image_size
from models.single_query_arch import SINGLE_QUERY_ARCH
from models.two_query_arch import TWO_QUERY_ARCH



def _lower_confidence_bound(NA: int, N: int, failure_prob: float) -> float:
    """ Returns a (1 - failure_prob) lower confidence bound on a bernoulli proportion.

    This function uses the Clopper-Pearson method.

    :param NA: the number of "successes"
    :param N: the number of total draws
    :param failure_prob: the confidence level
    :return: a lower bound on the binomial proportion which holds true w.p at least (1 - failure_prob) over the samples
    """
    return proportion_confint(NA, N, alpha=2 * failure_prob, method="beta")[0]

def certify(model, x, n_pred, n_cert, failure_prob, adv, image_size, batch_size):
    """
    """

    # assumes num_channels = 3
    image_dims = image_size * image_size * 3

    # get n_pred monte carlo samples for prediction
    counts_prediction = monte_carlo_predictions(model, x, n_pred, batch_size)
    prediction = counts_prediction.argmax().item()
    
    # get n_cert monte carlo samples for certification
    counts_estimation = monte_carlo_predictions(model, x, n_cert, batch_size)
    nA = counts_estimation[prediction].item()
    prob_lb = _lower_confidence_bound(nA, n_cert, failure_prob)

    if adv == "l1":
        pass
    elif adv == "l2":
        radius = conf.arch.total_sigma * norm.ppf(prob_lb)
    elif adv == "linf":
        l2_radius = conf.arch.total_sigma * norm.ppf(prob_lb)
        radius = l2_radius / np.sqrt(image_dims)

    if prob_lb < 0.5:
        prediction = -1 # abstain

    return prediction, radius, prob_lb

def _count_arr(arr, length):
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts

def monte_carlo_predictions(model, x, num, batch_size):
    """Sample the base classifier's prediction under noisy corruptions of the input x.

    :param x: the input [channel x width x height]
    :param num: number of samples to collect
    :param batch_size:
    :return: an ndarray[int] of length num_classes containing the per-class counts
    """

    with torch.no_grad():
        count_size = 2 if conf.data.num_classes == 1 else conf.data.num_classes
        counts = np.zeros(count_size, dtype=int)
        for _ in range(math.ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            input_batch = x.repeat((this_batch_size, 1, 1, 1))

            # get model predictions
            logging_trackers = {
                'mode': 'certify',
            }
            outputs = model(input_batch, logging_trackers)

            if conf.data.dataset == "celeba":
                predictions = (sigmoid(outputs) > 0.5).int()
            else:
                predictions = outputs.argmax(1)
            counts += _count_arr(predictions.cpu().numpy(), count_size)  
        return counts

if __name__=="__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-s", "--seed", type=int, default=42, help="random seed")
    argparser.add_argument("-y", "--yaml", help="paths to base configs. Loaded from left-to-right.", default=list())
    argparser.add_argument("--log_dir", type=str, default=None, help="log dir of the trained model that needs to be certified.")
    
    args = argparser.parse_args()
    conf = OmegaConf.load(args.yaml)
    # set the seed
    set_seed(args.seed)

    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        raise ValueError('must specify log_dir!')

    # prepare output file
    outfile = os.path.join(log_dir, "certification_log_"+str(conf.certify.n_cert)+".txt")
    f = open(outfile, 'w+')
    print("idx \t label \t predict \t "+ conf.certify.adv +"_radius \t prob_lb \t correct \t time", file=f, flush=True)

    # load test dataset
    transform_params = {
        "pad_size": conf.data.transform.pad_size,
        "num_image_locations": conf.data.transform.num_image_locations,
        "background": conf.data.transform.background,
        "mask_supervise": conf.trainer.mask_supervise,
        "face_feature": conf.data.transform.face_feature
    }

    test_dataset = get_dataset(dataset=conf.data.dataset,
                               split='test',
                               path=os.path.join(
                                   conf.data.dataset_path, conf.data.dataset),
                               transform_params=transform_params)

    # create model
    if conf.arch.num_query == 1:
        model = SINGLE_QUERY_ARCH(conf).to(conf.trainer.device)        
    else:
        model = TWO_QUERY_ARCH(conf).to(conf.trainer.device)
    # print(model)
        
    # get the checkpointed state
    model_sd_path = os.path.join(log_dir, "model_sd.pt")
    checkpoint = torch.load(model_sd_path)

    # load the saved model parameters
    model.load_state_dict(checkpoint)

    # get image size
    image_size_assumed = get_image_size(conf) 
    print("assumed image size is : ", image_size_assumed)
    image_size = test_dataset[0][0].shape[-1]
    print("real image size is : ", image_size)
    assert image_size == image_size_assumed
    # assert conf.trainer.mask_supervise == 0

    # main certify loop
    model.eval()
    for i in tqdm(range(len(test_dataset))):
        # only certify every skip examples, and stop after max examples
        if i % conf.certify.skip != 0:
            continue
        if i == conf.certify.max:
            break

        (x, label) = test_dataset[i]

        before_time = time.time()
        x = x.to(conf.trainer.device)

        # certify the prediction of g around x
        prediction, radius, prob_lb = certify(model,
                                              x,
                                              conf.certify.n_pred,
                                              conf.certify.n_cert, 
                                              conf.certify.failure_prob,
                                              conf.certify.adv,
                                              image_size,
                                              conf.certify.cert_batch_size) 
        after_time = time.time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{} \t {} \t {} \t {:.3} \t {:3} \t {} \t {}".format(
            i, label, prediction, radius, prob_lb, correct, time_elapsed), file=f, flush=True)
        # print("Certified input {}".format(i))

    f.close()
