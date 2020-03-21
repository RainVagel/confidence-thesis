#!/usr/bin/env bash

for i in {1..5}
do
echo "Start training LeNet on MNIST Mact $i"
python training.py 'MNIST' 'lenet' "exps_paper/mnist_mact_$i" 'True' 'mact' '100'
echo 'MNIST LeNet trained Mact'
done

for i in {1..5}
do
echo "Start training LeNet on MNIST Softmax $i"
python training.py 'MNIST' 'lenet' "exps_paper/mnist_softmax_$i" 'False' 'softmax' '100'
echo 'MNIST LeNet trained Softmax'
done