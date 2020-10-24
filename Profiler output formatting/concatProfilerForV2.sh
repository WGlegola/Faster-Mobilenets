#!/bin/bash
cd ../models/$1/profiler
for i in 1 2 3; do
    mkdir $i
    cat $i".txt" | grep Convolution | tr , . > $i/Convolution.txt
    cat $i".txt" | grep BatchNorm | tr , . > $i/BatchNorm.txt
    cat $i".txt" | grep clip | tr , . > $i/clip.txt
    cat $i".txt" | grep DeleteVariable | tr , . > $i/DeleteVariable.txt
    cat $i".txt" | grep elemwise_add | tr , . > $i/elemwise_add.txt
    cat $i".txt" | grep expand_dims | tr , . > $i/expand_dims.txt
    cat $i".txt" | grep Pooling | tr , . > $i/Pooling.txt
    cat $i".txt" | grep Flatten | tr , . > $i/Flatten.txt
done