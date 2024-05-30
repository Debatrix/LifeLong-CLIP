# Wraping for the Caltech101 dataset

import os
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets import Caltech101 as Caltech101_tv
from torchvision.transforms import transforms


class Caltech101(Caltech101_tv):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(
            root, "category",
            transforms.ToTensor() if transform is None else transform,
            target_transform, download)

        self.categories.remove("Faces_easy")
        self.categories.remove("caltech-101")

        name_map = {
            "airplanes": "airplane",
            "Faces": "face",
            "Leopards": "leopard",
            "Motorbikes": "motorbike",
        }

        self.annotation_categories = list(
            map(lambda x: name_map[x]
                if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(
                os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

        self.classes_names = self.categories

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )).convert('RGB')

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    ))
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
