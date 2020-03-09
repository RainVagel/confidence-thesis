#!/usr/bin/env bash

echo 'Start training LeNet on MNIST Mact'
python training.py 'MNIST' 'lenet' 'tf_upgrade' 'True' 'mact'
echo 'MNIST LeNet trained Mact'

echo 'Start training LeNet on MNIST Softmax'
python training.py 'MNIST' 'lenet' 'tf_upgrade' 'False' 'softmax'
echo 'MNIST LeNet trained Softmax'