#!/bin/bash

ALL=`ls -1 cifar10_models/state_dicts | grep -Po ".+(?=\.pt)"`
echo -e "  subset\t   equal\t    diff\t   total"
echo "---------------------------------------------------------"
for i in $ALL; do
  echo -e "$i;"
  python3 check_labels_probabilities.py --classifier $i --all --no_header
done
