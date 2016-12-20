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

#database = CIFAR10(batch_size = 100)
#data, labels = database.getTestSet(asBatches=False)
#print len(data), len(labels)
#convnet = ConvolutionalNetwork(database)
#params = convnet.get_params()
#gen = Genome(params)
#print gen

#n_iter = 10
#print "creating"
#indiv1 = Individual()
#print indiv1.fitness
#indiv2 = Individual()
#print "growing"
#indiv1.grow(n_iter)
#print indiv1.fitness
#print "crossover"
#ind3, ind4 = indiv1.crossover(indiv2, n_iter)
#print ind3.fitness
#print "growing child"
#ind3.grow(n_iter)
#print ind3.fitness
#print "mutating"
#ind3.mutate(1)
#print ind3.fitness
#print "done"


# TEST FOR INDIVIDUAL RECOVERY -> IS WORKING :D
#print "\n"*10,"Recovering individual test"
#indiv = Individual()
#indiv.grow_all_params(1000, first_iter=True)
#print "Fitness after 1000 iters %.2f"%(indiv.fitness)
#indiv.repair_and_grow(1000, repair_iters=400)
#print "\n"*10


fitness = list()
n_gen = 100
population = Population(n_indiv=6)
for i in range(n_gen):
    fitness.append(population.iter())

with open('workfile.pkl', 'w') as f:
    pickle.dump({'fitness': fitness}, f, pickle.HIGHEST_PROTOCOL)

