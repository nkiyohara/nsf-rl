#!/bin/bash
# Training script for Latent NSF on PushT task

set -e

uv run scripts/train_latent_nsf_pusht.py \
    --data-root data/random_dmp_npz \
    --epochs 10000 \
    --batch-size 64 \
    --lr 1e-3 \
    --seq-len 16 \
    --latent-dim 8 \
    --posterior-type gru \
    --flow-type affine_coupling \
    --wandb-project Latent-NSF-PushT
