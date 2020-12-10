#!/bin/bash

ALL=`ls -1 cifar10_models/state_dicts | ggrep -Po ".+(?=\.pt)"`
echo -e "model    \ttrain acc\ttest acc\ttotal acc"
echo "---------------------------------------------------------"
for i in $ALL; do
  echo -ne "$i\t"
  python3 compare_class_labels.py --classifier $i --data_dir dataset --no_header
done
