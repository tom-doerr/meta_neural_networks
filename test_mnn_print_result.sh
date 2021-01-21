#!/bin/bash

./test_mnn.sh | awk 'BEGIN{i=0} /^\{'\''Accuracy/ {print $0; i++;}'
