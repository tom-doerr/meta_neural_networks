#!/bin/bash

./test_gpu.sh | awk 'BEGIN{i=0} /^\{'\''Accuracy/ {print i " " $0; i++;}'
