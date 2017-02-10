import tensorflow as tf
import numpy as np
import cPickle as pickle

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer
from .model import NeuralNetwork, ConvolutionalNetwork
from .genetic import Genome, KernelChromosome, KernelGene
from .population import Individual, Population

'''[[(3, 3, 3, 32), (32,)], [(3, 3, 32, 32), (32,)], [(3, 3, 32, 64), (64,)], [(3, 3, 64, 64), (64,)], [(3, 3, 64, 128), (128,)], [(3, 3, 128, 128), (128,)], [(3, 3, 128, 128), (128,)]]
'''
fitness = []
son_props = []
n_gen = 100
population = Population(n_indiv=16)
for i in range(n_gen):
    (fit_iter, son_iter) = population.iter()
    fitness.append(fit_iter)
    son_props.append(son_iter)
    with open('deep-genet-log-single-point.pkl', 'w') as f:
        pickle.dump({'fitness': np.array(fitness),
                     'son_props': np.array(son_props)},
                    f, pickle.HIGHEST_PROTOCOL)

