#!/usr/bin/env bash

echo 'Start training LeNet on MNIST Mact'
python training.py 'MNIST' 'lenet' 'full_reg' 'True' 'mact' '100'
echo 'MNIST LeNet trained Mact'

echo 'Start training LeNet on MNIST Softmax'
python training.py 'MNIST' 'lenet' 'full_reg' 'False' 'softmax' '100'
echo 'MNIST LeNet trained Softmax'