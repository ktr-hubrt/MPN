#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON Train_meta.py --gpus $1 --dataset_path '../ano_pred_cvpr2018/Data/' --dataset_type $2 --exp_dir $3