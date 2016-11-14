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
        self.train_all_params = self.optimizer.minimize(self.loss)
        self.train_fc_params = self.optimizer.minimize(self.loss,
                                                       var_list=self.fc_params)
        self.correct_predictions = tf.equal(tf.argmax(self.model_output,1),
                                            tf.argmax(self.target,1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_predictions, tf.float32), name='accuracy')
        self.reset_params()
    
    def build_model(self):
        raise NotImplementedError("Must override NeuralNetwork.build_model()")

    def reset_params(self):
        self.sess.run(tf.initialize_all_variables())
        
    def train_iterations(self, n_iterations, just_fc=False):
        if just_fc:
            train_step = self.train_fc_params
        else:
            train_step = self.train_all_params
            
        for iteration in range(n_iterations):
            data, labels = self.database.nextBatch()
            _ = self.sess.run((train_step),feed_dict={
                self.model_input: data,
                self.target: labels,
                self.keep_prob: 0.5
            })
        train_accuracy = self.accuracy.eval(
            feed_dict={
                self.model_input: data,
                self.target: labels,
                self.keep_prob: 0.5# to be fair
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
                    self.target: labels,
                    self.keep_prob: 1.0
                    }, session=self.sess
                )
            )
        return np.array(accuracies).mean()

    def get_params(self):
        params = []
        for layer in self.conv_layers:
            params.append(layer.get_params(self.sess))
        return params

    def set_params(self, params):
        for param, layer in zip(params, self.conv_layers):
            layer.set_params(param, self.sess)
            
class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self, database):
        super(ConvolutionalNetwork, self).__init__(database)

    def build_model(self):
        # Temporal implementation
        self.model_input = tf.placeholder(tf.float32)
        self.target = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.conv_layer_1 = ConvolutionalLayer(self.model_input,
                                               [3, 3, 3, 32],
                                               'conv_layer_1')
        self.conv_layer_2 = ConvolutionalLayer(self.conv_layer_1.output_tensor,
                                               [3, 3, 32, 32],
                                               'conv_layer_2')

        self.pool_layer_3 = MaxPoolingLayer(self.conv_layer_2.output_tensor,
                                            'pool_layer_3')

        self.conv_layer_4 = ConvolutionalLayer(self.pool_layer_3.output_tensor,
                                               [3, 3, 32, 64],
                                               'conv_layer_4')
        self.conv_layer_5 = ConvolutionalLayer(self.conv_layer_4.output_tensor,
                                               [3, 3, 64, 64],
                                               'conv_layer_5')

        self.pool_layer_6 = MaxPoolingLayer(self.conv_layer_5.output_tensor,
                                            'pool_layer_6')

        self.conv_layer_7 = ConvolutionalLayer(self.pool_layer_6.output_tensor,
                                               [3, 3, 64, 128],
                                               'conv_layer_7')
        self.conv_layer_8 = ConvolutionalLayer(self.conv_layer_7.output_tensor,
                                               [3, 3, 128, 128],
                                               'conv_layer_8')
        self.conv_layer_9 = ConvolutionalLayer(self.conv_layer_8.output_tensor,
                                               [3, 3, 128, 128],
                                               'conv_layer_9')

        self.pool_layer_10 = MaxPoolingLayer(self.conv_layer_9.output_tensor,
                                            'pool_layer_10')

        self.fc_input = tf.reshape(self.pool_layer_10.output_tensor,
                                   [-1, 4*4*128])

        self.fc_layer_11 = FullyConnectedLayer(self.fc_input,
                                               [4*4*128, 100],
                                               'fc_layer_11')
        self.dropout_output_12 = tf.nn.dropout(self.fc_layer_11.output_tensor,
                                               self.keep_prob)
        self.fc_layer_13 = FullyConnectedLayer(self.dropout_output_12,
                                               [100, 10],
                                               'fc_layer_13')
        self.conv_layers = [self.conv_layer_1,
                            self.conv_layer_2,
                            self.conv_layer_4,
                            self.conv_layer_5,
                            self.conv_layer_7,
                            self.conv_layer_8,
                            self.conv_layer_9]

        self.fc_layers = [self.fc_layer_11,
                          self.fc_layer_13]

        self.fc_params = []
        for layer in self.fc_layers:
            self.fc_params.append(layer.weights)
            self.fc_params.append(layer.biases)
            
        self.model_output = self.fc_layer_13.output_tensor
        
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
    convnet.train_iterations(100, just_fc=True)
    print "Performance after 100 fc iterations", convnet.evaluate()
    params = convnet.get_params()
    print [[param.shape for param in params_pair] for params_pair in convnet.get_params()]
    zero_params = [[np.zeros(param.shape) for param in params_pair] for params_pair in convnet.get_params()]
    convnet.set_params(zero_params)
    print "Performance with zeros as params", convnet.evaluate()
    convnet.train_iterations(100)
    print "Performance after 100 iterations", convnet.evaluate()
    convnet.reset_params()
    print "Reseting params..."
    convnet.train_iterations(100)
    print "Performance after 100 iterations", convnet.evaluate()
    
