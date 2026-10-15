#!/usr/bin/env bash
# Component-level copying (Sec. 5.2): MLPCopy (INIT=ffncopy) or FullCopy (INIT=fullcopy) from a Softmax teacher.
set -e
DATA_DIR=${DATA_DIR:-./data}
TEACHER=${TEACHER:-./checkpoints/deit_small_patch16_224.pth}
MODEL=${MODEL:-vit_small_patch16_224}
OP=${OP:-relu}                               # elu1 | relu | inline | mhla | ttt
INIT=${INIT:-ffncopy}                        # ffncopy (MLPCopy) | fullcopy (FullCopy)
DATASET=${DATASET:-torch/cifar100}; NUM_CLASSES=${NUM_CLASSES:-100}
EPOCHS=${EPOCHS:-300}; WARMUP=${WARMUP:-50}
OUTPUT=${OUTPUT:-./output}; NPROC=${NPROC:-4}

if [ "${INIT}" = "fullcopy" ] && [ "${OP}" != "mhla" ] && [ "${OP}" != "ttt" ]; then
    LAYOUT="class_token=True no_embed_class=False sin_pe=False"; GP=token   # DeiT layout: class token + learned PE
else
    LAYOUT="class_token=False no_embed_class=True sin_pe=True"; GP=avg      # no class token, sinusoidal PE, avg pool
fi

torchrun --nproc_per_node=${NPROC} train.py \
    --data-dir ${DATA_DIR} --dataset ${DATASET} --dataset-download --num-classes ${NUM_CLASSES} \
    --input-size 3 224 224 --model ${MODEL} \
    --model-kwargs attn_type=${OP} weight_init=${INIT} copy_ckpt_path=${TEACHER} ${LAYOUT} --gp ${GP} \
    -b 128 -j 4 --opt adamw --lr 2e-3 --min-lr 1e-6 --weight-decay 0.05 --clip-grad 3.0 \
    --sched cosine --epochs ${EPOCHS} --warmup-epochs ${WARMUP} \
    --checkpoint-hist 1 --output ${OUTPUT} --experiment ${OP}_${INIT}_$(basename ${DATASET})
