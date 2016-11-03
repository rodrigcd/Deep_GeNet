import tensorflow as tf
import numpy as np

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, FullyConnectedLayer

class NeuralNetwork(object):
    def __init__(self, database):
        self.database = database
        # self.sess = tf.Session()

    def train_iterations(self, n_iterations):
        for iteration in range(n_iterations):
            data, labels = self.database.nextBatch()
            self.train_step.eval(feed_dict={
                self.model_input: data,
                self.target: labels
            })
        train_accuracy = self.accuracy.eval(
            feed_dict={
                self.model_input: data,
                self.target: labels
                }
            )
        return train_accuracy
            
    def update_genome(self, genome):
        # TODO
        pass

    def evaluate(self, use_test=False):
        if use_test:
            batches = self.database.getTestSet(asBatches=True)
        else:
            batches = self.database.getValidationSet(asBatches=True)
        accuracies = []
        for batch in batches:
            data, labels = batch
            accuracies.append(
                self.accuracy.eval(feed_dict={
                    self.model_input: data,
                    self.target: labels
                    }
                )
            )
        return np.array(accuracies).mean()
    
class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self, database):
        super(ConvolutionalNetwork, self).__init__(database)

class ResidualNetwork(NeuralNetwork):
    def __init__(self):
        super(ResidualNetwork, self).__init__(database)
    

if __name__=='__main__':
    database = CIFAR10(batch_size=100)
    data, labels = database.getTestSet()
    print len(data), len(labels)
