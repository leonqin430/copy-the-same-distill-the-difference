#!/usr/bin/env bash
# ImageNet-1K (Sec. 4.3 / Appendix C.3): any operator and initialization, 300 epochs, total batch 4096.
# Per-GPU batch and accumulation per model size (4 GPUs): tiny 512x2, small 256x4, base 64x16.
set -e
DATA_DIR=${DATA_DIR:-./data/imagenet}        # ImageFolder layout: train/<class>/*.JPEG, val/<class>/*.JPEG
TEACHER=${TEACHER:-./checkpoints/deit_small_patch16_224.pth}
MODEL=${MODEL:-vit_small_patch16_224}        # vit_tiny_patch16_224 | vit_small_patch16_224 | vit_base_patch16_224
OP=${OP:-relu}; INIT=${INIT:-ffncopy}
BS=${BS:-256}; ACCUM=${ACCUM:-4}
EPOCHS=${EPOCHS:-300}; WARMUP=${WARMUP:-50}
OUTPUT=${OUTPUT:-./output}; NPROC=${NPROC:-4}

KWARGS="attn_type=${OP} weight_init=${INIT} class_token=False no_embed_class=True sin_pe=True"
case "${INIT}" in copy|ffncopy|fullcopy) KWARGS="${KWARGS} copy_ckpt_path=${TEACHER}";; esac
DISTILL=""; [ -n "${LAMBDA}" ] && DISTILL="--distill-teacher ${TEACHER} --distill-lambda ${LAMBDA} --distill-loss mse_out"

torchrun --nproc_per_node=${NPROC} train.py \
    --data-dir ${DATA_DIR} --dataset imagenet --num-classes 1000 \
    --input-size 3 224 224 --model ${MODEL} --model-kwargs ${KWARGS} --gp avg ${DISTILL} \
    -b ${BS} --grad-accum-steps ${ACCUM} -j 8 --opt adamw --lr 2e-3 --min-lr 1e-6 --weight-decay 0.05 --clip-grad 3.0 \
    --sched cosine --epochs ${EPOCHS} --warmup-epochs ${WARMUP} \
    --checkpoint-hist 1 --output ${OUTPUT} --experiment ${OP}_${INIT}_imagenet
