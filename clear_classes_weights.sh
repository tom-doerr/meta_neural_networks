#!/bin/bash

ALL=`ls -1 cifar10_models/state_dicts | grep -Po ".+(?=\.pt$)"`
for i in $ALL; do
  echo -n "$i:"
  rm -r "cifar10_models/state_dicts/$i" &>/dev/null
  echo " class models deleted"  # echo newline
done
