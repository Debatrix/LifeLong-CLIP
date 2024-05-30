#!/bin/bash

GPUS=$1
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="adapter-clip"
PEFT_ENCODER='both' # both, text, image
DATASET="tinyimagenet" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

VIS_CLASS="batch"

RAND_NM="--rnd_NM"
# RAND_NM=""

ZS_TEST="--zero_shot_evaluation"
# ZS_TEST=""

MEM_SIZE=0

if [ "$DATASET" == "cifar100" ]; then
    ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"

elif [ "$DATASET" == "tinyimagenet" ]; then
    ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"

elif [ "$DATASET" == "imagenet-r" ]; then
    ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"

else
    echo "Undefined setting"
    exit 1
fi


for seed in 1 2 3 4 5
do
    INFO="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"
    CUDA_VISIBLE_DEVICES=${GPUS} python main.py --method $METHOD \
    --dataset $DATASET $ZS_TEST \
    --n_tasks $N_TASKS --m $M --n $N --rnd_NM \
    --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
    --note $INFO --eval_period $EVAL_PERIOD --n_worker 4 --num_gpus ${NB_GPUS}
done
