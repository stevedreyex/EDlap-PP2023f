#!/bin/bash

cd omp
make clean

cd ../gpu_version
make clean

cd ../omp_gpu_version
make clean

cd ../rule_of_thumb
./clean_rule_of_thumb.sh

cd ../
