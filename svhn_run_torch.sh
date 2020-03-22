#!/usr/bin/env bash

for i in {1..5};
do
echo "Start training ResNet on SVHN Mact $i";
python training.py 'SVHN' 'resnet' "exps_paper/svhn_mact_$i" 'True' 'mact' '100';
echo 'SVHN ResNet trained Mact';
done

for i in {1..5};
do
echo "Start training Resnet on SVHN Softmax $i";
python training.py 'SVHN' 'resnet' "exps_paper/svhn_softmax_$i" 'False' 'softmax' '100';
echo 'SVHN ResNet trained Softmax';
done