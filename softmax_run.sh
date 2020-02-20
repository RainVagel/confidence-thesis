#!/usr/bin/env bash

echo 'Start training resnet on SVHN'
python training.py 'SVHN' 'resnet' 'False'
echo 'SVHN resnet trained'

echo 'Start training resnet on cifar10'
python training.py 'CIFAR10' 'resnet' 'False'
echo 'CIFAR10 resent trained'

echo 'STart training resnet on cifar100'
python training.py 'CIFAR100' 'resnet' 'False'
echo 'CIFAR100 resnet trained'

echo 'Start training LeNet on MNIST'
python training.py 'MNIST' 'lenet' 'False'
echo 'MNIST LeNet trained'