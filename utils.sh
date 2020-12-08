#!/bin/bash

docker_run() {
    docker run -it --mount src="$(pwd)",target=/mounted,type=bind --gpus all meta_nn bash -ic " \
        cd /mounted/pytorch-cifar/; \
        $@"
}
