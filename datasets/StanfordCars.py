# Wraping for the StanfordCars dataset

from typing import Callable, Optional

from torchvision.datasets import StanfordCars as StanfordCars_tv
from torchvision.transforms import transforms


class StanfordCars(StanfordCars_tv):

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


if __name__ == "__main__":
    dataset = StanfordCars(root="data", train=False, download=True)
    print(len(dataset), len(dataset.classes_names))
