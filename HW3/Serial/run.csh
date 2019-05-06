#! /bin/csh -f

#compile
make

#run
make test N=192 seed=3
