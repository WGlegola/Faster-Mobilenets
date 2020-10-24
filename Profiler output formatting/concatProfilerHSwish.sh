#!/bin/bash
cd ../models/$1/profiler
declare -a arr=("Convolution"
"BatchNorm"
"clip"
"_div_scalar"
"broadcast_mul"
"_plus_scalar"
"FullyConnected"
"DeleteVariable"
"expand_dims"
"Pooling"
"Flatten")
for i in 1 2 3; do
    mkdir $i
    for y in "${arr[@]}"
    do
        cat $i".txt" | grep $y | tr , . > $i/$y.txt
    done
done








