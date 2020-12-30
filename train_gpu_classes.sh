#!/bin/bash
for i in {0..9}; do
    python3 cifar10_train.py --max_epochs 1 --learning_rate 0.000000000001 --classifier mobilenet_v2 --pretrained --data_dir dataset --target $i --gpus 0,
done
