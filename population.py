import tensorflow as tf
import numpy as np

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer
from .model import NeuralNetwork, ConvolutionalNetwork
from .genetic import Genome, KernelChromosome, KernelGene

def proportional_random_selection(fitness_list):
    accumulated_fitness = np.cumsum(fitness_list)
    accumulated_probs = 1.0*accumulated_fitness/accumulated_fitness[-1]
    random_num = np.random.uniform()
    for individual_idx, cum_prob in enumerate(accumulated_probs):
        if random_num < cum_prob:
            return individual_idx
    raise Exception('Bug on proportional_random_selection function.')

class Individual(object):
    'Basic indiviual with a genome and a phenotype'
    database = CIFAR10(batch_size = 100, augment_data=True)
    convnet = ConvolutionalNetwork(database)
    def __init__(self):
        self.convnet.reset_params()
        self.phenotype = self.convnet.get_params()
        self.genome = Genome(self.phenotype)
        self.fitness = 0

    def update_genome(self):
        self.genome.set_parameters(self.convnet.get_params())

    def update_phenotype(self):
        self.convnet.set_params(self.genome.get_parameters())

    def grow_all_params(self, n_iter, first_iter=False):
        if first_iter:
            self.convnet.reset_params()
        self.convnet.train_iterations(n_iter, just_fc=False)
        self.fitness = self.convnet.evaluate()
        print "[grow_all_params] Val. acc: %.2f"%(self.fitness)
        self.update_genome()
        
    def repair(self, repair_iters=400):
        self.convnet.reset_params()
        self.update_phenotype()
        self.convnet.train_iterations(repair_iters, just_fc=True)
        self.fitness = self.convnet.evaluate()
        print "[repair] Val. acc: %.2f"%(self.fitness)
        self.update_genome()
        
    def repair_and_grow(self, grow_iters, repair_iters=400):
        print "\nRepair and grow..."
        self.repair(repair_iters=repair_iters)
        self.grow_all_params(grow_iters)
        print "\n"

    def crossover(self, individual, repair_iters=400):
        child1 = Individual()
        child2 = Individual()
        genome1, genome2 = self.genome.crossover(individual.genome)
        child1.genome = genome1
        child2.genome = genome2
        child1.repair(repair_iters=repair_iters)
        child2.repair(repair_iters=repair_iters)
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
        self.n_iter = 1000
        self.fix_n_indiv = n_indiv
        for i in range(n_indiv):
            individual = Individual()
            individual.grow_all_params(100, first_iter=True)
            self.individuals.append(individual)
        print self.individuals
        self.update_fitness()

    def crossover(self):
        for i in range(int(self.n_indiv/2)):
            parent1_idx = proportional_random_selection(self.fitness)
            parent2_idx = parent1_idx
            while parent1_idx == parent2_idx:
                parent2_idx = proportional_random_selection(self.fitness)
            print "[Population.crossover] Crossing %d and %d"%(parent1_idx, parent2_idx)
            child1, child2 = self.individuals[parent1_idx].crossover(
                self.individuals[parent2_idx], self.n_iter)
            self.individuals.append(child1)
            self.individuals.append(child2)
            self.n_indiv += 2
        self.update_fitness()

    def mutate(self, p = 0.05):
        for i in range(self.n_indiv):
            self.individuals[i].mutate(p)

    def update_fitness(self):
        # ASSUMES THAT EACH INDIVIDUAL KEEPS AN UPDATED VERSION OF ITS FITNESS
        self.fitness = [individual.fitness for individual in self.individuals]

    def repair_and_grow(self):
        for individual in self.individuals:
            individual.repair_and_grow(self.n_iter, repair_iters=400)
        self.update_fitness()
        
    def select_individuals(self):
        index = np.argsort(self.fitness)
        index = index[-self.fix_n_indiv:]
        is_son = index>=(self.n_indiv//2)
        son_prop = is_son.astype('float32').mean()
        aux_list = list()
        self.individuals = [self.individuals[i] for i in index]
        self.n_indiv = len(self.individuals)
        self.update_fitness()
        return son_prop
        
    def print_statistics(self):
        print "Population fitness:\nMean: %.2f, Max: %.2f"%(np.mean(self.fitness), np.max(self.fitness))
        
    def iter(self):
        print("\nGeneration = %d\n"%(self.generation))
        self.print_statistics()
        
        print "Crossover..."
        self.crossover()
        self.print_statistics()
        
        print "Selection..."
        son_prop = self.select_individuals()
        print "Sons proportion %.2f"%(son_prop)
        self.print_statistics()
        
        print "Repair and grow..."
        self.repair_and_grow()
        
        self.generation += 1
        return self.fitness, son_prop
