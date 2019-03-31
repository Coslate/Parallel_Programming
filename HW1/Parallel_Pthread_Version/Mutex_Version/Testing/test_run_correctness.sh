#! /bin/bash

loop_num=$1
j=0

rm -rf ./test_run.log
while [ $j -lt $loop_num ]
do
    echo "loop $j ..."
    make test &>> test_run.log 
    (( j++ ))
done

