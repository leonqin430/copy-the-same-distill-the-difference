#!/usr/bin/env bash
# Softmax recovery target: the pre-trained Softmax ViT (DeiT) fine-tuned on the same task with the same recipe.
set -e
DATA_DIR=${DATA_DIR:-./data}
TEACHER=${TEACHER:-./checkpoints/deit_small_patch16_224.pth}
MODEL=${MODEL:-vit_small_patch16_224}
DATASET=${DATASET:-torch/cifar100}; NUM_CLASSES=${NUM_CLASSES:-100}
EPOCHS=${EPOCHS:-300}; WARMUP=${WARMUP:-50}
OUTPUT=${OUTPUT:-./output}; NPROC=${NPROC:-4}

torchrun --nproc_per_node=${NPROC} train.py \
    --data-dir ${DATA_DIR} --dataset ${DATASET} --dataset-download --num-classes ${NUM_CLASSES} \
    --input-size 3 224 224 --model ${MODEL} \
    --model-kwargs attn_type=softmax weight_init=fullcopy copy_ckpt_path=${TEACHER} class_token=True no_embed_class=False sin_pe=False \
    --gp token -b 128 -j 4 --opt adamw --lr 2e-3 --min-lr 1e-6 --weight-decay 0.05 --clip-grad 3.0 \
    --sched cosine --epochs ${EPOCHS} --warmup-epochs ${WARMUP} \
    --checkpoint-hist 1 --output ${OUTPUT} --experiment softmax_$(basename ${DATASET})
