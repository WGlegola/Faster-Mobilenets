#!/bin/bash
cd models/$1
rm profiler -r
mkdir profiler
for i in 1 2 3; do
    echo 1 $i 0
    python3 test.py --model $1 --saved-params $2 --dtype float32 --input-pic /home/ubuntu/magisterka/data/val/0 > profiler/$i".txt"
    rm profile.json
    for ((ii=1;ii<=49;ii++)); do
        echo 1 $i $ii
        python3 test.py --model $1 --saved-params $2 --dtype float32 --input-pic /home/ubuntu/magisterka/data/val/$ii >> profiler/$i".txt"
        rm profile.json
    done
done
