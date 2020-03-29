#!/usr/bin/env bash

for i in {1..5};
do
echo "Start training ResNet on CIFAR100 Mact $i";
python training.py 'CIFAR100' 'resnet' "exps_paper/cifar100_mact_$i" 'True' 'mact' '100';
echo 'CIFAR100 ResNet trained Mact';
done

for i in {1..5};
do
echo "Start training Resnet on CIFAR100 Softmax $i";
python training.py 'CIFAR100' 'resnet' "exps_paper/cifar100_softmax_$i" 'False' 'softmax' '100';
echo 'CIFAR100 ResNet trained Softmax';
done