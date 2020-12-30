#!/bin/bash
python3 cifar10_train.py --max_epochs 1 --classifier mobilenet_v2 --data_dir dataset --gpus 0,
