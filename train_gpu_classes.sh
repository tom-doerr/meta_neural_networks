#!/bin/bash
for i in {0..9}; do
    python3 cifar10_train.py --max_epochs 50 --classifier mobilenet_v2 --pretrained --data_dir dataset --target $i --gpus 0,
    # exit;
done
