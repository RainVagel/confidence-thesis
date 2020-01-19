#!/usr/bin/env bash

# Will run 10 experiments and save the models using both tanh and relu activation functions.
for i in {1..11}; do
  python training.py 'create_tanh_model' "four_moons/tanh_$i" 'd100_d100_d100_d100_d2_tanh_MActAbs'
  python training.py 'create_relu_model' "four_moons/relu_$i" 'd100_d100_d100_d100_d2_tanh_MActAbs'
done
