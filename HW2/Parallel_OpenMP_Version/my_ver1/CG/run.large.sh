#! /bin/sh

make clean
make DATASIZE=LARGE

taskset -c 0 ./bin/cg
taskset -c 0,1 ./bin/cg
taskset -c 0,1,2,3 ./bin/cg
