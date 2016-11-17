import tensorflow as tf
import numpy as np

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer
from .model import NeuralNetwork, ConvolutionalNetwork
from .genetic import Genome, KernelChromosome, KernelGene
from .population import Individual

'''[[(3, 3, 3, 32), (32,)], [(3, 3, 32, 32), (32,)], [(3, 3, 32, 64), (64,)], [(3, 3, 64, 64), (64,)], [(3, 3, 64, 128), (128,)], [(3, 3, 128, 128), (128,)], [(3, 3, 128, 128), (128,)]]
'''

#database = CIFAR10(batch_size = 100)
#data, labels = database.getTestSet(asBatches=False)
#print len(data), len(labels)
#convnet = ConvolutionalNetwork(database)
#params = convnet.get_params()
#gen = Genome(params)
#print gen

n_iter = 10
print "creating"
indiv1 = Individual()
print indiv1.fitness
indiv2 = Individual()
print "growing"
indiv1.grow(n_iter)
print indiv1.fitness
print "crossover"
ind3, ind4 = indiv1.crossover(indiv2)
print ind3.fitness
print "growing child"
ind3.grow(n_iter)
ind3.fitness
print "done"
