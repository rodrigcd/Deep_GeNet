import tensorflow as tf

class Layer(object):
    def __init__(self, input_tensor, layer_name):
        self.input_tensor = input_tensor
        self.layer_name = layer_name
        # A self.output_tensor should be implemented on inherited classes

class ConvolutionalLayer(Layer):
    def __init__(self, input_tensor, kernel_shape, layer_name):
        super(ConvolutionalLayer, self).__init__(input_tensor, layer_name)
        self.kernel_shape = kernel_shape
        with tf.variable_scope(layer_name):
            self.weights = tf.get_variable("weights", self.kernel_shape,
                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.biases = tf.get_variable("biases", [self.kernel_shape[3]],
                                          initializer=tf.constant_initializer(0.0))
            self.conv_out = tf.nn.conv2d(self.input_tensor, self.weights,
                                         strides=[1,1,1,1], padding='SAME')
            self.output_tensor = tf.nn.relu(self.conv_out + self.biases)

class FullyConnectedLayer(Layer):
    def __init__(self, input_tensor, weights_shape, layer_name):
        super(FullyConnectedLayer, self).__init__(input_tensor, layer_name)
        self.weights_shape = weights_shape
        with tf.variable_scope(layer_name):
            self.weights = tf.get_variable("weights", self.weights_shape,
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.biases = tf.get_variable("biases", [self.weights_shape[1]],
                                          initializer=tf.constant_initializer(0.0))
            self.mult_out = tf.matmul(self.input_tensor, self.weights)
            self.output_tensor = tf.nn.relu(self.mult_out + self.biases)

class MaxPoolingLayer(Layer):
    def __init__(self, input_tensor, layer_name,
                 ksize=[1,2,2,1],
                 strides=[1,2,2,1]):
        super(MaxPoolingLayer, self).__init__(input_tensor, layer_name)
        self.output_tensor = tf.nn.max_pool(self.input_tensor,
                                            ksize=ksize,
                                            strides=strides,
                                            padding='SAME',
                                            name=self.layer_name)
        
