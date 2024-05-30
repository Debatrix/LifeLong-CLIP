#!/bin/bash

GPUS=$1
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

# Short description of the experiment.
NOTE="zs"

METHOD="maple"
DATASET="tinyimagenet" # cifar100, tinyimagenet, imagenet-r
N_TASKS=5
N=50
M=10

MEM_SIZE=0
VIS_CLASS="batch"

ZS_TEST="--zero_shot_evaluation"
# ZS_TEST=""

GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

if [ "$DATASET" == "cifar100" ]; then
    ONLINE_ITER=3    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-4 OPT_NAME="sgd" SCHED_NAME="default"

elif [ "$DATASET" == "tinyimagenet" ]; then
    ONLINE_ITER=3    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-4 OPT_NAME="sgd" SCHED_NAME="default"

elif [ "$DATASET" == "imagenet-r" ]; then
    ONLINE_ITER=3    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-4 OPT_NAME="sgd" SCHED_NAME="default"

else
    echo "Undefined setting"
    exit 1
fi

for PEFT_ENCODER in both text image
do
    for VIS_CLASS in batch all
    do
        for seed in 1 2 3 4 5
        do
            INFO="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py --method $METHOD \
            --dataset $DATASET $ZS_TEST\
            --n_tasks $N_TASKS --m $M --n $N --rnd_NM \
            --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
            --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
            --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
            --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
            --note $INFO --eval_period $EVAL_PERIOD --n_worker 4 --num_gpus ${NB_GPUS}
        done
    done
done
