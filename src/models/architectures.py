import torch
import torch.backends.cudnn as cudnn
from torchvision.models.resnet import resnet50
from multiquery_randomized_smoothing.src.models.cifar_resnet import resnet as resnet_cifar
from multiquery_randomized_smoothing.src.dataset import get_normalize_layer, PreProcessLayer
from multiquery_randomized_smoothing.src.models.unet import UNet

ARCHITECTURES = ["resnet50", "cifar_resnet110"]

def get_architecture(conf: dict,
                     arch: str = "cifar_resnet110",
                     prepend_preprocess_layer: bool = False, 
                     prepend_normalize_layer: bool = False,
                     dataset: str = "cifar10",
                     input_size: int = 32, # required for when we adjust in_features of fc layer of cifar_resnet110
                     num_classes: int = 10) -> torch.nn.Module:
    """Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above 
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    # for classifier
    if dataset == "cifar10" and arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, input_size=input_size, num_classes=num_classes)
    elif dataset == "celeba" and arch == "resnet50":
        model = resnet50(weights=None, num_classes=num_classes)
    elif dataset == "imagenet" and arch == "resnet50": 
        # model = torch.nn.DataParallel(resnet50(pretrained=False))
        # cudnn.benchmark = True
        model = resnet50(weights=None)
    # for mask model
    elif arch == "unet":
        model = UNet(in_channels=conf.unet.in_channels, 
                     out_channels=conf.unet.out_channels,
                     channel=conf.unet.channel)

    if prepend_normalize_layer:
        normalize_layer = get_normalize_layer(dataset, input_size)
        model = torch.nn.Sequential(normalize_layer, model)

    if prepend_preprocess_layer:
        preprocess_layer = PreProcessLayer(prob_flip=0.5)
        model = torch.nn.Sequential(preprocess_layer, model)

    return model
