from typing import Callable, Optional
import os

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

# TinyImageNet dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR


class Imagenet_R(ImageFolder):

    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:

        self.root = os.path.expanduser(root)
        self.url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
        self.filename = 'imagenet-r.tar'

        if not os.path.exists(os.path.join(self.root, 'imagenet-r')):
            fpath = os.path.join(self.root, self.filename)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError(
                        'Dataset not found. You can use download=True to download it'
                    )
                else:
                    print('Downloading from ' + self.url)
                    download_url(self.url, self.root, filename=self.filename)
            import tarfile
            tar = tarfile.open(fpath, 'r')
            tar.extractall(os.path.join(self.root))
            tar.close()

        self.path = self.root + '/imagenet-r/'
        super().__init__(self.path,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.RandomCrop(224)
                         ]) if transform is None else transforms.Compose([
                             transforms.Resize(256),
                             transforms.RandomCrop(224), transform
                         ]),
                         target_transform=target_transform)
        generator = torch.Generator().manual_seed(0)
        len_train = int(len(self.samples) * 0.8)
        len_test = len(self.samples) - len_train
        self.train_sample = torch.randperm(len(self.samples),
                                           generator=generator)
        self.test_sample = self.train_sample[len_train:].sort().values.tolist()
        self.train_sample = self.train_sample[:len_train].sort().values.tolist(
        )

        if train:
            self.classes = [i for i in range(200)]
            self.class_to_idx = [i for i in range(200)]
            samples = []
            for idx in self.train_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

        else:
            self.classes = [i for i in range(200)]
            self.class_to_idx = [i for i in range(200)]
            samples = []
            for idx in self.test_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

        self.classes_names = [
            'goldfish', 'great_white_shark', 'hammerhead', 'stingray', 'hen',
            'ostrich', 'goldfinch', 'junco', 'bald_eagle', 'vulture', 'newt',
            'axolotl', 'tree_frog', 'iguana', 'African_chameleon', 'cobra',
            'scorpion', 'tarantula', 'centipede', 'peacock', 'lorikeet',
            'hummingbird', 'toucan', 'duck', 'goose', 'black_swan', 'koala',
            'jellyfish', 'snail', 'lobster', 'hermit_crab', 'flamingo',
            'american_egret', 'pelican', 'king_penguin', 'grey_whale',
            'killer_whale', 'sea_lion', 'chihuahua', 'shih_tzu',
            'afghan_hound', 'basset_hound', 'beagle', 'bloodhound',
            'italian_greyhound', 'whippet', 'weimaraner', 'yorkshire_terrier',
            'boston_terrier', 'scottish_terrier',
            'west_highland_white_terrier', 'golden_retriever',
            'labrador_retriever', 'cocker_spaniels', 'collie', 'border_collie',
            'rottweiler', 'german_shepherd_dog', 'boxer', 'french_bulldog',
            'saint_bernard', 'husky', 'dalmatian', 'pug', 'pomeranian',
            'chow_chow', 'pembroke_welsh_corgi', 'toy_poodle',
            'standard_poodle', 'timber_wolf', 'hyena', 'red_fox', 'tabby_cat',
            'leopard', 'snow_leopard', 'lion', 'tiger', 'cheetah',
            'polar_bear', 'meerkat', 'ladybug', 'fly', 'bee', 'ant',
            'grasshopper', 'cockroach', 'mantis', 'dragonfly',
            'monarch_butterfly', 'starfish', 'wood_rabbit', 'porcupine',
            'fox_squirrel', 'beaver', 'guinea_pig', 'zebra', 'pig',
            'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 'badger',
            'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'panda',
            'eel', 'clown_fish', 'puffer_fish', 'accordion', 'ambulance',
            'assault_rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball',
            'bathtub', 'lighthouse', 'beer_glass', 'binoculars', 'birdhouse',
            'bow_tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon',
            'canoe', 'carousel', 'castle', 'mobile_phone', 'cowboy_hat',
            'electric_guitar', 'fire_engine', 'flute', 'gasmask',
            'grand_piano', 'guillotine', 'hammer', 'harmonica', 'harp',
            'hatchet', 'jeep', 'joystick', 'lab_coat', 'lawn_mower',
            'lipstick', 'mailbox', 'missile', 'mitten', 'parachute',
            'pickup_truck', 'pirate_ship', 'revolver', 'rugby_ball', 'sandal',
            'saxophone', 'school_bus', 'schooner', 'shield', 'soccer_ball',
            'space_shuttle', 'spider_web', 'steam_locomotive', 'scarf',
            'submarine', 'tank', 'tennis_ball', 'tractor', 'trombone', 'vase',
            'violin', 'military_aircraft', 'wine_bottle', 'ice_cream', 'bagel',
            'pretzel', 'cheeseburger', 'hotdog', 'cabbage', 'broccoli',
            'cucumber', 'bell_pepper', 'mushroom', 'Granny_Smith',
            'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate',
            'pizza', 'burrito', 'espresso', 'volcano', 'baseball_player',
            'scuba_diver', 'acorn'
        ]

    def __len__(self) -> int:
        return len(self.samples)
