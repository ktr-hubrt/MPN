#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON Test_meta.py --gpus $1 --dataset_path '../ano_pred_cvpr2018/Data/' --dataset_type $2 --model_dir 'exp/'$2'/'$3
