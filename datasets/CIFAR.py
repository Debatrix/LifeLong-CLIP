# Wraping for the CIFAR dataset

from typing import Callable, Optional

from torchvision.datasets import CIFAR10 as CIFAR10_tv
from torchvision.datasets import CIFAR100 as CIFAR100_tv
from torchvision.transforms import transforms


class CIFAR10(CIFAR10_tv):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(
            root, "train" if train else "test",
            transforms.ToTensor() if transform is None else transform,
            target_transform, download)

        self.classes_names = self.classes


class CIFAR100(CIFAR100_tv):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(
            root, "train" if train else "test",
            transforms.ToTensor() if transform is None else transform,
            target_transform, download)

        self.classes_names = self.classes
