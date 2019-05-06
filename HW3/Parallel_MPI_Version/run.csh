#! /bin/csh -f

#compile
make

#run
make test n=4 N=192 seed=3
