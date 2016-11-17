import tensorflow as tf
import numpy as np

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer
from .model import NeuralNetwork, ConvolutionalNetwork
from .genetic import Genome, KernelChromosome, KernelGene

class Individual(object):
    'Basic indiviual with a genome and a phenotype'
    database = CIFAR10(batch_size = 100, augment_data=True)
    convnet = ConvolutionalNetwork(database)
    def __init__(self):
        self.phenotype = self.convnet.get_params()
        self.genome = Genome(self.phenotype)
        self.fitness = 0

    def update_genome(self):
        self.genome.set_parameters(self.phenotype)

    def update_phenotype(self):
        self.convnet.set_params(self.genome.get_parameters())

    def grow(self, n_iter, just_fc = False):
        self.update_phenotype()
        self.convnet.train_iterations(n_iter, just_fc = just_fc)
        self.fitness = self.convnet.evaluate()
        self.update_genome()

    def crossover(self, individual, n_iter):
        child1 = Individual()
        child2 = Individual()
        genome1, genome2 = self.genome.crossover(individual.genome)
        child1.genome = genome1
        child2.genome = genome2
        child1.grow(int(n_iter/2), just_fc = True)
        child2.grow(int(n_iter/2), just_fc = True)
        child1.grow(int(n_iter/2))
        child2.grow(int(n_iter/2))
        return child1, child2

    def mutate(self, p):
        self.genome.mutate(p)

    def evaluate(self):
        self.update_phenotype()
        self.fitness = self.convnet.evaluate()
        self.update_genome()
        return self.fitness


class Population(object):
    'Main object to control genetic algorithm'
    def __init__(self, n_indiv = 10, selection_method = 'proportional'):
        self.individuals = list()
        self.fitness = list()
        self.n_indiv = n_indiv
        self.generation = 0
        self.n_iter = 50
        self.fix_n_indiv = n_indiv
        aux_fitness = np.zeros(n_indiv)
        for i in range(n_indiv):
            self.individuals.append(Individual())
            aux_fitness[i] = self.individuals[i].evaluate()
            self.individuals[i].convnet.reset_params

        self.fitness = aux_fitness

    def crossover(self):
        acumulated_prob = np.cumsum(fitness[self.generation])
        acumulated_prob = acumulated_prob/np.max(acumulated_prob)
        aux = self.n_indiv
        for i in range(int(self.n_indiv/2)):
            random1 = np.random.uniform()
            random2 = np.random.uniform()
            index1 = np.min(np.nonzero(np.divide_floor(acumulated_prob,random1)))
            index2 = np.min(np.nonzero(np.divide_floor(acumulated_prob,random2)))
            child1, child2 = self.individuals[index1].crossover(self.individuals[index2], self.n_iter)
            self.individuals.append(child1)
            self.individuals.append(child2)
            aux+= 2
        self.n_indiv = aux

    def mutate(self, p = 0.05):
        for i in range(self.n_indiv):
            self.individuals[i].mutate(p)

    def update_fitness(self):
        aux_fitness = np.zeros(self.n_indiv)
        for i in range(self.n_indiv):
            aux_fitness[i] = self.individuals[i].evaluate()
        self.fitness = aux_fitness

    def grow_population(self):
        for i in range(self.n_indiv):
            self.individuals[i].grow(self.n_iter)
            self.fitness[i] = self.individuals[i].evaluate()

    def select_individuals(self):
        index = np.argsort(self.fitness)
        index = index[-self.fix_n_indiv:]
        aux_list = list()
        for i in index:
            aux_list.append(self.individuals[i])
        self.individuals = aux_list

    def iter(self):
        self.grow_population()
        self.crossover()
        self.update_fitness()
        self.select_individuals()
        self.mutate()
        self.generation += 1
        















