#!/bin/bash

set -x 
source utils.sh

#if [[ ! -d pytorch-cifar/ ]]
#then
#    git clone git@github.com:kuangliu/pytorch-cifar.git
#fi

docker_run " \
    ./train_gpu.sh
"
