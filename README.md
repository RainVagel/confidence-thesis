# Radial Softmax: A Novel Activation Function for Neural Networks to Reduce Overconfidence in Out-Of-Distribution Data
This repository hosts the code for the respective thesis. Feel free to run tests for yourself and use the radial softmax layer.

## Description
This repository hosts the code for the thesis. It has the following:
1. Radial Softmax layer implemented in Tensorflow 2.0 and usable with the Keras functional and Sequence API-s. This layer is implemented in the models.py file.
2. Experimentation Jupyter notebook to train models on the moons datasets.
3. Code to train the Small Resnet and LeNet-5 models on SVHN, MNIST, CIFAR-10 and CIFAR-100 datasets.
4. Code to analyse trained models with out-of-distribution datasets.
5. Code to visualise class scores and probabilities.

## Using the code
This section will explain how to use the provided code.

### Moons dataset
The moons_thesis.ipynb file was used to generate the moons models and their confidence plots. The architecture is already there, the user just has to choose if they want to use radial softmax or regular softmax and the number of iterations as well as the dataset. In the end it will output the model files, zoomed-in and zoomed-out area plots and metrics.

### Extra analysis figures for moons models
The file areas_analyser.py was used to get extra figures for the moons datasets. This file does not accept command line arguments and the user must input the location of the model files into the lists at the end of the file.

### Training Small Resnet or LeNet-5
To train either of the files you can use the command line. For example to train a LeNet model with MNIST:
```
python training.py 'MNIST' 'lenet' 'paper_extra/mnist_mact' 'True' 'mact' '100' ''
```

The commands are the following:
 1. 'MNIST' - Specifies the dataset. Either MNIST, CIFAR-100, CIFAR-10 or SVHN
 2. 'lenet' - Specifies the model type, either 'lenet' or 'resnet'
 3. 'paper_extra/mnist_mact' - Specifies the output folder
 4. 'True' - Either to use radial softmax or regular softmax. True for radial, false for regular
 5. 'mact' - Additional name for the output files.
 6. '100' - Number of iterations to train the model for.
 7. '' - Leave empty for ReLU models, input either 'leakyrelu' or 'selu' to use those activations for the later layers.

### Analysing LeNet-5 or Small Resnet
The code to use for analysing the LeNet-5 or Small Resnet models are in analysis.py. It can also be used through the command line.
```
python analysis.py 'compute' 'model_folder' 
```

The commands are the following:
1. 'compute' - Wether to use the analysis or compute functionality. Analysis creates analysis files by running out-of-distribution data through the models. Compute gets the metrics from those files.
2. 'model_folder' - The folder where the necessary files are in.
