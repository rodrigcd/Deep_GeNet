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
        self.convnet.reset_params()
        self.phenotype = self.convnet.get_params()
        self.genome = Genome(self.phenotype)
        self.fitness = 0

    def update_genome(self):
        self.genome.set_parameters(self.convnet.get_params())

    def update_phenotype(self):
        self.convnet.set_params(self.genome.get_parameters())

    def grow_all_params(self, n_iter, first_iter=False):#, just_fc = False):
        #self.convnet.reset_params()
        #self.update_phenotype()
        #print "[grow] before 300 iters of fc tuning %f"%(self.convnet.evaluate())
        #self.convnet.train_iterations(300, just_fc=True)
        #print "[grow] after fc tuning %f"%(self.convnet.evaluate())
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
        #child1.grow(int(n_iter/2), just_fc = True)
        #child2.grow(int(n_iter/2), just_fc = True)
        #child1.grow(int(n_iter/2))
        #child2.grow(int(n_iter/2))
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
            individual.grow_all_params(600, first_iter=True)
            self.individuals.append(individual)
        print self.individuals
        self.update_fitness()

    def crossover(self):
        acumulated_prob = np.cumsum(self.fitness)
        acumulated_prob = acumulated_prob/np.max(acumulated_prob)
        aux = self.n_indiv
        for i in range(int(self.n_indiv/2)):
            random1 = np.random.uniform()
            random2 = np.random.uniform()
            index1 = np.min(np.nonzero(np.floor_divide(acumulated_prob,random1)))
            index2 = np.min(np.nonzero(np.floor_divide(acumulated_prob,random2)))
            print "[Population.crossover] Crossing %d and %d"%(index1, index2)
            child1, child2 = self.individuals[index1].crossover(self.individuals[index2], self.n_iter)
            self.individuals.append(child1)
            self.individuals.append(child2)
            aux += 2
        self.n_indiv = aux
        self.update_fitness()

    def mutate(self, p = 0.05):
        for i in range(self.n_indiv):
            self.individuals[i].mutate(p)

    def update_fitness(self):
        #aux_fitness = np.zeros(self.n_indiv)
        #for i in range(self.n_indiv):
        #    aux_fitness[i] = self.individuals[i].evaluate()
        #self.fitness = aux_fitness

        # ASSUMES THAT EACH INDIVIDUAL KEEPS AN UPDATED VERSION OF ITS FITNESS
        self.fitness = [individual.fitness for individual in self.individuals]

    #def grow_population(self):
    #    for i in range(self.n_indiv):
    #        self.individuals[i].grow(self.n_iter)
    #    self.update_fitness()

    def repair_and_grow(self):
        for individual in self.individuals:
            individual.repair_and_grow(self.n_iter, repair_iters=400)
        self.update_fitness()
        
    def select_individuals(self):
        index = np.argsort(self.fitness)
        index = index[-self.fix_n_indiv:]
        aux_list = list()
        for i in index:
            aux_list.append(self.individuals[i])
        self.individuals = aux_list
        self.n_indiv = len(self.individuals)
        self.update_fitness()
        
    def print_statistics(self):
        print "Population fitness:\nMean: %.2f, Max: %.2f"%(np.mean(self.fitness), np.max(self.fitness))
        
    def iter(self):
        print("\nGeneration = %d\n"%(self.generation))
        self.print_statistics()
        
        print "Crossover..."
        self.crossover()
        self.print_statistics()
        
        print "Selection..."
        self.select_individuals()
        self.print_statistics()
        
        print "Repair and grow..."
        self.repair_and_grow()
        
        self.generation += 1
        #print("\t growing")
        #self.grow_population()
        #print("\t Fitness after growing")
        #print(self.fitness)
        #print("\t best_fitness = "+str(np.max(self.fitness))+" mean_fitness = "+str(np.mean(self.fitness)))
        #print("\n crossover")
        #self.crossover()
        ##self.update_fitness()
        #print("best_fitness = "+str(np.max(self.fitness))+" mean_fitness = "+str(np.mean(self.fitness)))
        #self.select_individuals()
        #print(self.fitness)
        ##self.mutate()
        #self.generation += 1
        ##self.update_fitness
        #print("after selection and mutation")
        #print("best_fitness = "+str(np.max(self.fitness))+" mean_fitness = "+str(np.mean(self.fitness)))
        return self.fitness















