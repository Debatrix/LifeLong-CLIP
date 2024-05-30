#!/bin/bash

GPUS=$1
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="batch" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="mvp-clip"
DATASET="tinyimagenet" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=10
N=50
M=10

VIS_CLASS="batch"

GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-4 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-4 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-4 OPT_NAME="adam" SCHED_NAME="default"

else
    echo "Undefined setting"
    exit 1
fi


for seed in 1 2 3 4 5
do
    INFO="${METHOD}_${NOTE}_SEED${seed}"
    CUDA_VISIBLE_DEVICES=${GPUS} python main.py --method $METHOD \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $seed  --visible_classes $VIS_CLASS \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir data \
    --note $INFO --eval_period $EVAL_PERIOD --n_worker 4 --num_gpus ${NB_GPUS} --rnd_NM \
    --use_mask --use_contrastiv 
done
