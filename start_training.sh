#!/bin/bash

docker build docker -t meta_nn

if [[ ! -d pytorch-cifar/ ]]
then
    git clone git@github.com:kuangliu/pytorch-cifar.git
fi

docker run -it --mount src="$(pwd)",target=/mounted,type=bind --gpus all meta_nn bash -ic " \
    cd /mounted/pytorch-cifar/; \
    python3 main.py
"
