# Wraping for the Food101 dataset

from typing import Callable, Optional

from torchvision.datasets import Food101 as Food101_tv
from torchvision.transforms import transforms


class Food101(Food101_tv):

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
