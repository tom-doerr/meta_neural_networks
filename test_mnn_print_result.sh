#!/bin/bash

./test_mnn.sh $1 | awk 'BEGIN{i=0} /^\{'\''Accuracy/ {print $0; i++;}'
