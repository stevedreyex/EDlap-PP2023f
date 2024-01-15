#!/bin/bash

for dir in *; do
  if [ -d "$dir" ]; then
    cd $dir
    echo -n "Now testing the performance of " && basename `pwd`
    make cuda
    cd ../
  fi
done
