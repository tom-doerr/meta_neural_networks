#!/bin/bash
for i in {0..9}; do
    python3 cifar10_train.py --max_epochs 1 --classifier mobilenet_v2 --pretrained --data_dir dataset --target $i --gpus 0, --use_switch_func --switch_threshold 0.25
    # exit;
done
