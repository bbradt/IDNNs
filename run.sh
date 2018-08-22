#!/bin/bash

declare -a learning_rates=(0.0001)

for i in "${learning_rates[@]}"
do
    echo "LEARNING RATE"
    echo $i
    python main.py -learning_rate $i -num_epochs 8000 -num_repeat 1 -l1_reg 0.0 -l2_reg 0.0
done

