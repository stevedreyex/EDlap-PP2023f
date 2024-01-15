#!/bin/bash

for dir in *; do
  if [ -d "$dir" ]; then
    cd $dir
    make cuda
    cd ../
  fi
done
