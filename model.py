import tensorflow as tf
import numpy as np

from .data import Database

class NeuralNetwork(object):
    def __init__(self, database):
        self.database = database

    def train_iterations(self, n_iterations):
        # TODO
        pass    

    def update_genome(self, genome):
        # TODO
        pass

    def evaluate(self):
        # TODO
        pass
    
class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self, database):
        super(ConvolutionalNetwork, self).__init__(database)

class ResidualNetwork(NeuralNetwork):
    def __init__(self):
        super(ResidualNetwork, self).__init__(database)
    
