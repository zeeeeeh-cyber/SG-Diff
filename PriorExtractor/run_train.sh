#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 train.py \
  --img_dir datasets/artifactIma \
  --mask_dir dataProcessing/dataset_mask \
  --sdf_dir  dataProcessing/dataset_sdf \
  --split_dir datasets/splits \
  --fold_idx 0 \
  --epochs 15000 --batch_size 4 --lr 0.0001 --lambda_sdf 1.0 \
  --checkpoint_dir ./checkpoints

