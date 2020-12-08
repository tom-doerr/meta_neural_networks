#!/bin/bash

set -x 
source utils.sh

docker build docker -t meta_nn

if [[ ! -d pytorch-cifar/ ]]
then
    git clone git@github.com:kuangliu/pytorch-cifar.git
fi

docker_run " \
    python3 main.py
"
