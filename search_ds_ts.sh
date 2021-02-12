#!/bin/bash
for ds in {10..20}; do
    echo "dataset switch_thr=0.$ds"
    ./train_classes_from_scratch_print_result.sh 0.$ds

    for ts in {10..30}; do
	echo "test switch_thr=0.$ts"
	./test_mnn_print_result.sh 0.$ts
    done;

done;

# Results can be processed with
# find max number and corresponding row
# cat -n test_search_ds_ts.txt | perl -nle '/test.*/ && ($t=1) || /(\d+).*?(\d+\.\d+).*/ && $t==1 && print($1, " ", $2) && ($t=0)' | sort -n -k2
# Print table "dataset switch_threshold rows | test switch_threshold cols"
# cat -n test_search_ds_ts.txt | perl -ne '/dataset.*?(\d+\.\d+)/ && print("\n", $1) || /test.*/ && ($t=1) || /(\d+).*?(\d+\.\d+).*/ && $t==1 && print(" ", $2) && ($t=0)'; echo ""
