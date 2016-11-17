import tensorflow as tf
import numpy as np

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer
from .model import NeuralNetwork, ConvolutionalNetwork
from .genetic import Genome, KernelChromosome, KernelGene

class Individual(object):
    'Basic indiviual with a genome and a phenotype'
    database = CIFAR10(batch_size = 100)
    convnet = ConvolutionalNetwork(database)
    def __init__(self):
        self.phenotype = self.convnet.get_params()
        self.genome = Genome(self.phenotype)
        self.fitness = 0

    def update_genome(self):
        self.genome.set_parameters(self.phenotype)

    def update_phenotype(self):
        convnet.set_params(self.genome.get_params)

    def grow(self, n_iter):
        self.update_phenotype
        self.convnet.train_iterations(n_iter)
        self.fitness = self.convnet.evaluate()
        self.update_genome

    def crossover(self, individual):
        child1 = Individual()
        child2 = Individual()
        genome1, genome2 = self.genome.crossover(individual.genome)
        child1.genome = genome1
        child2.genome = genome2
        return child1, child2



