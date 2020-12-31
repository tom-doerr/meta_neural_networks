#!/bin/bash

./test_gpu_classes.sh | awk 'BEGIN{i=0} /^\{/ {print i " " $0; i++;}'
