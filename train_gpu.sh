#!/bin/bash
python3 imagenet_train.py --max_epochs 1 --classifier mobilenet_v2 --data_dir /hdd --gpus 0,
