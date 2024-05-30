from typing import Callable, Optional
import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

# TinyImageNet dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR


class TinyImageNet(ImageFolder):

    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:

        self.root = os.path.expanduser(root)
        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        if not os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')):
            fpath = os.path.join(self.root, self.filename)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError(
                        'Dataset not found. You can use download=True to download it'
                    )
                else:
                    print('Downloading from ' + self.url)
                    download_url(self.url, self.root, filename=self.filename)
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(os.path.join(self.root))
            zip_ref.close()
            # self.split()

        self.path = self.root + '/tiny-imagenet-200/'
        if train:
            super().__init__(self.path + "train",
                             transform=transforms.ToTensor()
                             if transform is None else transform,
                             target_transform=target_transform)
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {
                clss: idx
                for idx, clss in enumerate(self.classes)
            }
            self.targets = []
            for idx, (path, _) in enumerate(self.samples):
                self.samples[idx] = (path,
                                     self.class_to_idx[path.split("/")[-3]])
                self.targets.append(self.class_to_idx[path.split("/")[-3]])

        else:
            super().__init__(self.path + "val",
                             transform=transforms.ToTensor()
                             if transform is None else transform,
                             target_transform=target_transform)
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {
                clss: idx
                for idx, clss in enumerate(self.classes)
            }
            self.targets = []
            with open(self.path + "val/val_annotations.txt", 'r') as f:
                file_to_idx = {
                    line.split('\t')[0]: self.class_to_idx[line.split('\t')[1]]
                    for line in f.readlines()
                }
                for idx, (path, _) in enumerate(self.samples):
                    self.samples[idx] = (path,
                                         file_to_idx[path.split("/")[-1]])
                    self.targets.append(file_to_idx[path.split("/")[-1]])

        self.classes_names = [
            "Egyptian Mau", "fishing casting reel", "volleyball",
            "rocking chair", "lemon", "American bullfrog", "basketball",
            "cliff", "espresso", "plunger", "parking meter",
            "German Shepherd Dog", "dining table", "monarch butterfly",
            "brown bear", "school bus", "pizza", "guinea pig", "umbrella",
            "pipe organ", "oboe", "maypole", "goldfish", "pot pie",
            "hourglass", "beach", "computer keyboard", "arabian camel",
            "ice cream", "metal nail", "space heater", "cardigan", "baboon",
            "snail", "coral reef", "albatross", "spider web", "sea cucumber",
            "backpack", "Labrador Retriever", "pretzel", "king penguin",
            "sulphur butterfly", "tarantula", "red panda", "soda bottle",
            "banana", "sock", "cockroach", "missile", "beer bottle",
            "praying mantis", "freight car", "guacamole", "remote control",
            "fire salamander", "lakeshore", "chimpanzee", "payphone",
            "fur coat", "mountain", "lampshade", "torch", "abacus",
            "moving van", "barrel", "tabby cat", "goose", "koala",
            "high-speed train", "CD player", "teapot", "birdhouse", "gazelle",
            "academic gown", "tractor", "ladybug", "miniskirt",
            "Golden Retriever", "triumphal arch", "cannon", "neck brace",
            "sombrero", "gas mask or respirator", "candle", "desk",
            "frying pan", "bee", "dam", "spiny lobster", "police van", "iPod",
            "punching bag", "lighthouse", "jellyfish", "wok", "potter's wheel",
            "sandal", "pill bottle", "butcher shop", "slug", "pig", "cougar",
            "construction crane", "vestment", "dragonfly",
            "automated teller machine", "mushroom", "rickshaw", "water tower",
            "storage chest", "snorkel", "sunglasses", "fly", "limousine",
            "black stork", "dugong", "sports car", "water jug",
            "suspension bridge", "ox", "popsicle", "turnstile",
            "Christmas stocking", "broom", "scorpion", "wooden spoon",
            "picket fence", "rugby ball", "sewing machine",
            "through arch bridge", "Persian cat", "refrigerator", "barn",
            "apron", "Yorkshire Terrier", "swim trunks / shorts", "stopwatch",
            "lawn mower", "thatched roof", "fountain", "southern black widow",
            "bikini", "plate", "teddy bear", "barbershop", "candy store",
            "station wagon", "scoreboard", "orange", "flagpole",
            "American lobster", "trolleybus", "drumstick", "dumbbell",
            "brass memorial plaque", "bow tie", "convertible", "bighorn sheep",
            "orangutan", "American alligator", "centipede", "syringe",
            "go-kart", "brain coral", "sea slug", "cliff dwelling",
            "mashed potatoes", "viaduct", "military uniform", "pomegranate",
            "chain", "kimono", "comic book", "trilobite", "bison", "pole",
            "boa constrictor", "poncho", "bathtub", "grasshopper",
            "stick insect", "Chihuahua", "tailed frog", "lion", "altar",
            "obelisk", "beaker", "bell pepper", "baluster / handrail",
            "bucket", "magnetic compass", "meatloaf", "gondola",
            "Standard Poodle", "acorn", "lifeboat", "binoculars",
            "cauliflower", "African bush elephant"
        ]
