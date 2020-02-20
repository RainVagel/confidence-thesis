#!/usr/bin/env bash

echo 'Start training resnet on SVHN'
python training.py 'SVHN' 'resnet' 'False' 'softmax'
echo 'SVHN resnet trained'

echo 'Start training resnet on cifar10'
python training.py 'CIFAR10' 'resnet' 'False' 'softmax'
echo 'CIFAR10 resent trained'

echo 'STart training resnet on cifar100'
python training.py 'CIFAR100' 'resnet' 'False' 'softmax'
echo 'CIFAR100 resnet trained'

echo 'Start training LeNet on MNIST'
python training.py 'MNIST' 'lenet' 'False' 'softmax'
echo 'MNIST LeNet trained'