# CLIP model is an Efficient Online Lifelong Learner

## Introduction

Official repository for online class incremental learning on stochastic blurry task boundary via mask and visual prompt tuning. Code is based on [Si-Blurry](https://github.com/naver-ai/i-Blurry) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning). If you use our method or code in your research, please consider citing the paper as follows:

```
@article{wang2024clip,
  title={CLIP model is an Efficient Online Lifelong Learner},
  author={Wang, Leyuan and Xiang, Liuyu and Wei, Yujie and Wang, Yunlong and He, Zhaofeng},
  journal={arXiv preprint arXiv:2405.15155},
  year={2024}
}
```

## Requirements

- Pytorch
- timm

or you can install a conda environment with :

```Bash
   conda env create -f environment.yml
```

## Run

```Bash
   bash scripts/lora_clip.sh
```
