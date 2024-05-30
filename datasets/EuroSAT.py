# Wraping for the EuroSAT dataset

from typing import Callable, Optional

from torchvision.datasets import EuroSAT as EuroSAT_tv
from torchvision.transforms import transforms


class EuroSAT(EuroSAT_tv):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(
            root,
            transforms.ToTensor() if transform is None else transform,
            target_transform, download)

        self.classes_names = self.classes
