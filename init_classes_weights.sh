#!/bin/bash

# ALL=`ls -1 cifar10_models/state_dicts | grep -Po ".+(?=\.pt$)"`
ALL=`ls -1 cifar10_models/state_dicts | grep -Po "mobilenet_v2(?=\.pt$)"`
for i in $ALL; do
  echo -n "$i:"
  rm -r "cifar10_models/state_dicts/$i" &>/dev/null
  mkdir "cifar10_models/state_dicts/$i"

  for j in {0..9}; do
    echo -n " $j"
    cp "cifar10_models/state_dicts/$i.pt" "cifar10_models/state_dicts/$i/$j.pt"
  done
  echo ""  # echo newline
done
