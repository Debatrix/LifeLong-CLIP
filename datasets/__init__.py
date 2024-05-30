from .OxfordIIITPet import OxfordIIITPet
from .EuroSAT import EuroSAT
from .Caltech101 import Caltech101
from .Food101 import Food101
from .CIFAR import CIFAR10, CIFAR100
from .CUB200 import CUB200
from .multiDatasets import multiDatasets
from .Flowers102 import Flowers102
from .NotMNIST import NotMNIST
from .SVHN import SVHN
from .TinyImageNet import TinyImageNet
from .MNIST import MNIST
from .FashionMNIST import FashionMNIST
from .Imagenet_R import Imagenet_R
from torchvision.datasets import ImageNet

__all__ = [
    "CUB200",
    "multiDatasets",
    "Flowers102",
    "NotMNIST",
    "SVHN",
    "TinyImageNet",
    "CIFAR10",
    "CIFAR100",
    "MNIST",
    "FashionMNIST",
    "ImageNet",
    "Imagenet_R",
    "Food101",
    "Caltech101",
]

# dictionary of tuple of dataset, mean, std
datasets = {
    "cifar10":
    (CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100":
    (CIFAR100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100),
    "svhn": (SVHN, (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970), 10),
    "fashionmnist": (FashionMNIST, (0.2860, ), (0.3530, ), 10),
    "mnist": (MNIST, (0.1307, ), (0.3081, ), 10),
    "tinyimagenet":
    (TinyImageNet, (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262), 200),
    "notmnist": (NotMNIST, (0.1307, ), (0.3081, ), 10),
    "cub200": (CUB200, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "imagenet": (ImageNet, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1000),
    "imagenet-r":
    (Imagenet_R, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "food101": (Food101, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 101),
    "caltech101": (Caltech101, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                   100),
    "eurosat": (EuroSAT, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 10),
    "flowers102": (Flowers102, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                   102),
    "oxford_pet": (OxfordIIITPet, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                   37),
}


def get_dataset(name):
    return datasets[name]
