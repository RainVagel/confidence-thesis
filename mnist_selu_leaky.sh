#!/usr/bin/env bash

echo "Start training LeNet on MNIST Mact SeLU";
python training.py 'MNIST' 'lenet' "paper_extra/mnist_selu_mact" 'True' 'mact' '100' 'selu';


echo "Start training LeNet on MNIST Mact LeakyReLU";
python training.py 'MNIST' 'lenet' "paper_extra/mnist_leakyrelu_mact" 'True' 'mact' '100' 'leakyrelu';

echo "Start training LeNet on MNIST Softmax SeLU";
python training.py 'MNIST' 'lenet' "paper_extra/mnist_selu_softmax" 'False' 'softmax' '100' 'selu';


echo "Start training LeNet on MNIST softmax LeakyReLU";
python training.py 'MNIST' 'lenet' "paper_extra/mnist_leakyrelu_softmax" 'False' 'softmax' '100' 'leakyrelu';