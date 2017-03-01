import tensorflow as tf
import numpy as np
import cPickle as pickle

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer
from .model import NeuralNetwork, ConvolutionalNetwork
#from .genetic import Genome, KernelChromosome, KernelGene
#from .population import Individual, Population

'''[[(3, 3, 3, 32), (32,)], [(3, 3, 32, 32), (32,)], [(3, 3, 32, 64), (64,)], [(3, 3, 64, 64), (64,)], [(3, 3, 64, 128), (128,)], [(3, 3, 128, 128), (128,)], [(3, 3, 128, 128), (128,)]]
'''

database = CIFAR10(batch_size=100, augment_data=True)
convnet = ConvolutionalNetwork(database)
for i in range(1500):
    print "Iter. %d"%(i*100)
    convnet.train_iterations(100)
    if i%10==0:
        val_acc = convnet.evaluate()
        train_loss = convnet.evaluate_loss('training')
        val_loss = convnet.evaluate_loss('validation')
        print "Iteration %d. Val. acc: %.2f, Train loss: %.2f, Val. loss: %.2f"%(
            i*100,
            val_acc,
            train_loss,
            val_loss)
