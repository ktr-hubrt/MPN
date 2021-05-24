#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON Train_meta.py --gpus $1 --dataset_path './Data/' --dataset_type $2 --exp_dir $3
