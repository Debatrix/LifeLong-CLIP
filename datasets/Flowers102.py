# Wraping for the SHVN dataset

import os
import json
from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets import Flowers102 as tv_Flowers102
from torchvision.transforms import transforms


class Flowers102(tv_Flowers102):

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

        with open(
                os.path.join(self._base_folder,
                             'split_zhou_OxfordFlowers.json'), 'r') as f:
            dataset_split = json.load(f)['test']

        classes_names = {x[1]: x[2] for x in dataset_split}

        self.classes_names = [
            classes_names[x] for x in range(max(self._labels) + 1)
        ]
