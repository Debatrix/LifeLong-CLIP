# Wraping for the OxfordIIITPet dataset

import os
import json
from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet as tv_OxfordIIITPet
from torchvision.transforms import transforms


class OxfordIIITPet(tv_OxfordIIITPet):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(
            root, "trainval", "category", None,
            transforms.ToTensor() if transform is None else transform,
            target_transform, download)

        mode = 'train' if train else 'val'

        with open(
                os.path.join(self._base_folder, 'split_zhou_OxfordPets.json'),
                'r') as f:
            dataset_split = json.load(f)[mode]

        img_list = set(x[0] for x in dataset_split)
        _images, _labels = [], []
        for idx in range(len(self._images)):
            if os.path.basename(self._images[idx]) in img_list:
                _images.append(self._images[idx])
                _labels.append(self._labels[idx])

        self._images, self._labels = _images, _labels

        classes_names = {x[1]: x[2] for x in dataset_split}

        self.classes_names = [
            classes_names[x] for x in range(max(self._labels) + 1)
        ]
