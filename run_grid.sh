#!/bin/bash

declare -a learning_rates=(0.001 0.01 0.05 0.1 0.2 0.5)

for i in "${learning_rates[@]}"
do
    python main.py -learning_rate $i
done
