# Deep-GeNet

This is the code for the project Deep-GeNet, which trains a convolutional neural network with a novel strategy that mixes genetic algorithms and gradient descent.

Required software/libraries:
-python2
-numpy
-tensorflow 0.12 (runs with some deprecation warnings)
-cPickle
-matplotlib (for plotting)

IMPORTANT: You must have a copy of CIFAR10 dataset. Automatic download is possible using a flag at CIFAR10 class constructor. Fix DIR_BINARIES value at data.py according to your configuration.

To run the code execute the following command from the console on the parent directory, where "Deep-GeNet" is the name of the folder that contains the code.
>> python2 -m Deep-GeNet.demo

This command will execute the Deep-GeNet algorithm and it will save a pickle file with the fitness of the population and the proportion of sons for each generation.

A traditional gradient descent training can be done using the following command from the parent directory:
>> python2 -m Deep-GeNet.simple_training
