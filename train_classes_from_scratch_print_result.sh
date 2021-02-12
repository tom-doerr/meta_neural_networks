#!/bin/bash
./clear_classes_weights.sh
./init_classes_weights.sh
./train_gpu_classes.sh $1 | awk 'BEGIN{i=0} /^\{'\''Accuracy/ {print $0; i++;}'
