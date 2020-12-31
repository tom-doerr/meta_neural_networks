#!/bin/bash
for i in {0..9}; do
    python3 cifar10_test.py --max_epochs 1 --classifier mobilenet_v2 --data_dir dataset --target $i --gpus 0,
done
