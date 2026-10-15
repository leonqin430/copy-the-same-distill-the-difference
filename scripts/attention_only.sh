#!/usr/bin/env bash
# Attention-only initializations (Sec. 4): Random / Mimetic / Impulse / Copy on a transfer dataset.
# INIT: skip (Random) | mimetic0.7_0.7 (Mimetic) | impulse3 (Impulse) | copy (Copy, needs TEACHER)
set -e
DATA_DIR=${DATA_DIR:-./data}
TEACHER=${TEACHER:-./checkpoints/deit_small_patch16_224.pth}
MODEL=${MODEL:-vit_small_patch16_224}        # vit_tiny_patch16_224 | vit_small_patch16_224
OP=${OP:-relu}                               # elu1 | relu | inline | mhla | ttt
INIT=${INIT:-copy}
DATASET=${DATASET:-torch/cifar100}; NUM_CLASSES=${NUM_CLASSES:-100}
EPOCHS=${EPOCHS:-300}; WARMUP=${WARMUP:-50}  # Flowers-102 / Pets: 600 / 100
OUTPUT=${OUTPUT:-./output}; NPROC=${NPROC:-4}

KWARGS="attn_type=${OP} weight_init=${INIT} class_token=False no_embed_class=True sin_pe=True"
[ "${INIT}" = "copy" ] && KWARGS="${KWARGS} copy_ckpt_path=${TEACHER}"

torchrun --nproc_per_node=${NPROC} train.py \
    --data-dir ${DATA_DIR} --dataset ${DATASET} --dataset-download --num-classes ${NUM_CLASSES} \
    --input-size 3 224 224 --model ${MODEL} --model-kwargs ${KWARGS} --gp avg \
    -b 128 -j 4 --opt adamw --lr 2e-3 --min-lr 1e-6 --weight-decay 0.05 --clip-grad 3.0 \
    --sched cosine --epochs ${EPOCHS} --warmup-epochs ${WARMUP} \
    --checkpoint-hist 1 --output ${OUTPUT} --experiment ${OP}_${INIT}_$(basename ${DATASET})
