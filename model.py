import tensorflow as tf
import numpy as np

from .data import Database, CIFAR10
from .layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer

class NeuralNetwork(object):
    def __init__(self, database):
        self.database = database
        self.sess = tf.Session()
        self.build_model()
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_step = self.optimizer.minimize(self.loss)
        self.correct_predictions = tf.equal(tf.argmax(self.model_output,1),
                                            tf.argmax(self.target,1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_predictions, tf.float32), name='accuracy')
        self.reset_params()
    
    def build_model(self):
        raise NotImplementedError("Must override NeuralNetwork.build_model()")

    def reset_params(self):
        self.sess.run(tf.initialize_all_variables())
        
    def train_iterations(self, n_iterations):
        for iteration in range(n_iterations):
            data, labels = self.database.nextBatch()
            _ = self.sess.run((self.train_step),feed_dict={
                self.model_input: data,
                self.target: labels
            })
        train_accuracy = self.accuracy.eval(
            feed_dict={
                self.model_input: data,
                self.target: labels
                }, session=self.sess
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
                    }, session=self.sess
                )
            )
        return np.array(accuracies).mean()

    def get_params(self):
        params = []
        for layer in self.layers:
            params.append(layer.get_params(self.sess))
        return params

    def set_params(self, params):
        for param, layer in zip(params, self.layers):
            layer.set_params(param, self.sess)
            
class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self, database):
        super(ConvolutionalNetwork, self).__init__(database)

    def build_model(self):
        # Temporal implementation
        self.model_input = tf.placeholder(tf.float32)
        self.target = tf.placeholder(tf.float32)
        self.conv_layer_1 = ConvolutionalLayer(self.model_input,
                                               [5, 5, 3, 32],
                                               'conv_layer_1')
        self.fc_input = tf.reshape(self.conv_layer_1.output_tensor,
                                   [-1, 32*32*32])
        self.fc_layer = FullyConnectedLayer(self.fc_input,
                                            [32*32*32, 10],
                                            'fc_layer_1')
        self.layers = [self.conv_layer_1,
                       self.fc_layer]
        self.model_output = self.fc_layer.output_tensor
        
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                self.model_output,
                self.target,
                name='loss')
            )
        
class ResidualNetwork(NeuralNetwork):
    def __init__(self):
        super(ResidualNetwork, self).__init__(database)
    

if __name__=='__main__':
    database = CIFAR10(batch_size=100)
    data, labels = database.getTestSet(asBatches=False)
    print len(data), len(labels)
    convnet = ConvolutionalNetwork(database)
    print "Initial performance", convnet.evaluate()
    convnet.train_iterations(100)
    print "Performance after 100 iterations", convnet.evaluate()
    params = convnet.get_params()
    print [[param.shape for param in params_pair] for params_pair in convnet.get_params()]
    zero_params = [[np.zeros(param.shape) for param in params_pair] for params_pair in convnet.get_params()]
    convnet.set_params(zero_params)
    print "Performance with zeros as params", convnet.evaluate()
    convnet.train_iterations(100)
    print "Performance after 100 iterations", convnet.evaluate()
