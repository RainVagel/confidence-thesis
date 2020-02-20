#!/usr/bin/env bash

# Will run 10 experiments and save the models using both tanh and relu activation functions.
#for i in {1..5}; do
#  python training.py 'create_tanh_model' "four_moons/tanh_modded_$i" 'd100_d100_d100_d100_d2_tanh_MActAbs'
#  python training.py 'create_relu_model' "four_moons/relu_modded_$i" 'd100_d100_d100_d100_d2_relu_MActAbs'
#done

echo 'Start training resnet on SVHN'
python training.py 'SVHN' 'resnet'
echo 'SVHN resnet trained'

echo 'Start training resnet on cifar10'
python training.py 'CIFAR10' 'resnet'
echo 'CIFAR10 resent trained'

echo 'STart training resnet on cifar100'
python training.py 'CIFAR100' 'resnet'
echo 'CIFAR100 resnet trained'

echo 'Start training LeNet on MNIST'
python training.py 'MNIST' 'lenet'
echo 'MNIST LeNet trained'
