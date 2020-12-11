#!/bin/bash
python3 generate_class_labels.py --classifier mobilenet_v2 --data_dir dataset --probabilities --all --gpus 0,
