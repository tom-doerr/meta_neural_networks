#!/bin/bash

RESULTS_FILE="plotting/switch_threshold_accuracy"
RESULTS_FILE_TMP="$RESULTS_FILE"_tmp

#while read p; do
#    #echo ${${p#*: }%"}"*} >> $RESULTS_FILE
#    a=${p#*: }
#    echo ${a%"}"} >> $RESULTS_FILE
#done < $RESULTS_FILE_TMP
#exit 


rm $RESULTS_FILE
rm $RESULTS_FILE_TMP
for threshold in $(seq 0.1 0.01 0.3)
do
    python3 mnn_test.py --classifier mobilenet_v2 --data_dir dataset --gpus 0 --switch_threshold $threshold | grep 'Accuracy' | tee -a $RESULTS_FILE_TMP
    printf "$threshold, " >> $RESULTS_FILE
    tail_results_tmp=$(tail -n1 $RESULTS_FILE_TMP)
    a=${tail_results_tmp#*: }
    echo ${a%"}"} >> $RESULTS_FILE
done
#while read p; do
#    #echo ${${p#*: }%"}"*} >> $RESULTS_FILE
#    a=${p#*: }
#    echo ${a%"}"} >> $RESULTS_FILE
#done < $RESULTS_FILE_TMP
