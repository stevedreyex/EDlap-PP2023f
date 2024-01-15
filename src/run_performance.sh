#!/bin/bash

cd omp
make

cd ../gpu_version
make

cd ../omp_gpu_version
make

cd ../rule_of_thumb
./run_rule_of_thumb.sh

cd ../
