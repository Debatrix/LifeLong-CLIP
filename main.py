# import torch
import time
from configuration import config
from datasets import *
from methods.adapter_clip import AdapterCLIP
from methods.er_baseline import ER
from methods.clib import CLIB
from methods.maple import MaPLe
from methods.mvp_clip import CLIP_MVP
from methods.rainbow_memory import RM
from methods.finetuning import FT
from methods.ewcpp import EWCpp
from methods.lwf import LwF
from methods.mvp import MVP
from methods.continual_clip import ContinualCLIP

# torch.backends.cudnn.enabled = False
methods = {
    "er": ER,
    "clib": CLIB,
    "rm": RM,
    "lwf": LwF,
    "Finetuning": FT,
    "ewc++": EWCpp,
    "mvp": MVP,
    "continual-clip": ContinualCLIP,
    "mvp-clip": CLIP_MVP,
    "maple": MaPLe,
    "adapter-clip": AdapterCLIP,
    "lora-clip": AdapterCLIP
}


def main():
    # Get Configurations
    args = config.base_parser()
    trainer = methods[args.method](**vars(args))

    trainer.run()


if __name__ == "__main__":
    main()
