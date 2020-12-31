#!/bin/bash

docker_run() {
    docker build docker -t meta_nn
    docker run -it --mount src="$(pwd)",target=/mounted,type=bind --gpus all meta_nn bash -ic " \
        cd /mounted/; \
        $@"
}
