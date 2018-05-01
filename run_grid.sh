#!/bin/bash

declare -a learning_rates=(0.0004 0.0005)
declare -a l1=(0.0001 0.0005 0.001 0.005 0.01 0.05)

for i in "${learning_rates[@]}"
do
    echo "LEARNING RATE"
    echo $i
    python main.py -learning_rate $i -num_epochs 3000 -num_repeat 10
done

for i in "${l1[@]}"
do
    echo "L2 REGULARIZATION"
    echo $i
    python main.py -l2 $i -num_epochs 3000 -num_repeat 10
done

for i in "${l1[@]}"
do
    echo "ELASTIC NET REGULARIZATION"
    echo $i
    python main.py -l1 $i -l2 $i -num_epochs 3000 -num_repeat 10
done
