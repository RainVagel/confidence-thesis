#!/usr/bin/env bash

# Will run 10 experiments and save the models using both tanh and relu activation functions.
#for i in {1..5}; do
#  python training.py 'create_tanh_model' "four_moons/tanh_modded_$i" 'd100_d100_d100_d100_d2_tanh_MActAbs'
#  python training.py 'create_relu_model' "four_moons/relu_modded_$i" 'd100_d100_d100_d100_d2_relu_MActAbs'
#done

python training.py 'create_tanh_model' 500 "four_moons/tanh_z_plots" 'd100_d100_d100_d100_d2_tanh_MActAbs'
