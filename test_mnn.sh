#!/bin/bash
python3 mnn_test.py --classifier mobilenet_v2 --data_dir dataset --gpus 0, --switch_threshold 0.18
