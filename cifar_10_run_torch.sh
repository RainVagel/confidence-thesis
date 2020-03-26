#!/usr/bin/env bash

for i in {1..5};
do
echo "Start training ResNet on CIFAR10 Mact $i";
python training.py 'CIFAR10' 'resnet' "exps_paper/cifar10_mact_$i" 'True' 'mact' '100';
echo 'CIFAR10 ResNet trained Mact';
done

for i in {1..5};
do
echo "Start training Resnet on CIFAR10 Softmax $i";
python training.py 'CIFAR10' 'resnet' "exps_paper/cifar10_softmax_$i" 'False' 'softmax' '100';
echo 'CIFAR10 ResNet trained Softmax';
done