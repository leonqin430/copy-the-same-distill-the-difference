#!/usr/bin/env bash
# Attention distillation from a frozen Softmax teacher (Sec. 5.1 / 5.3): any initialization "w/ Distill".
# INIT: skip (Random) | copy (Copy) | ffncopy (MLPCopy) | fullcopy (FullCopy)
# LOSS: mse_out (paper default: MSE on attention outputs) | mse_map | ce_map (loss ablation, Sec. 5.4)
set -e
DATA_DIR=${DATA_DIR:-./data}
TEACHER=${TEACHER:-./checkpoints/deit_small_patch16_224.pth}
MODEL=${MODEL:-vit_small_patch16_224}
OP=${OP:-relu}                               # elu1 | relu | inline | mhla | ttt
INIT=${INIT:-copy}
LAMBDA=${LAMBDA:-50}; LOSS=${LOSS:-mse_out}
DATASET=${DATASET:-torch/cifar100}; NUM_CLASSES=${NUM_CLASSES:-100}
EPOCHS=${EPOCHS:-300}; WARMUP=${WARMUP:-50}
OUTPUT=${OUTPUT:-./output}; NPROC=${NPROC:-4}

if [ "${OP}" = "mhla" ] || [ "${OP}" = "ttt" ]; then
    LAYOUT="class_token=False no_embed_class=True sin_pe=True"; GP=avg      # square token grid, no class token
else
    LAYOUT="class_token=True no_embed_class=False sin_pe=False"; GP=token   # DeiT layout
fi
KWARGS="attn_type=${OP} weight_init=${INIT} ${LAYOUT}"
[ "${INIT}" != "skip" ] && KWARGS="${KWARGS} copy_ckpt_path=${TEACHER}"

torchrun --nproc_per_node=${NPROC} train.py \
    --data-dir ${DATA_DIR} --dataset ${DATASET} --dataset-download --num-classes ${NUM_CLASSES} \
    --input-size 3 224 224 --model ${MODEL} --model-kwargs ${KWARGS} --gp ${GP} \
    --distill-teacher ${TEACHER} --distill-lambda ${LAMBDA} --distill-loss ${LOSS} \
    -b 128 -j 4 --opt adamw --lr 2e-3 --min-lr 1e-6 --weight-decay 0.05 --clip-grad 3.0 \
    --sched cosine --epochs ${EPOCHS} --warmup-epochs ${WARMUP} \
    --checkpoint-hist 1 --output ${OUTPUT} --experiment ${OP}_${INIT}_distill_$(basename ${DATASET})
