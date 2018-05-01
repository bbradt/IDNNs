#!/bin/bash

declare -a activations=(6 5 4 3 2 1)

for i in "${activations[@]}"
do
    echo "ACTIVATION"
    echo $i
    python main.py -activation_function $i -num_epochs 3000 -num_repeat 10
done
