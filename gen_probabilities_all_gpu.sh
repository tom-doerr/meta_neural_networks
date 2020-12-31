#!/bin/bash

ALL=`ls -1 cifar10_models/state_dicts | grep -Po ".+(?=\.pt$)"`
for i in $ALL; do
  echo -e "$i"
  python3 generate_class_labels.py --classifier $i --data_dir dataset --probabilities --all --gpus 0,
  echo "---------------------------------------------------------"
done
