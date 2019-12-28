import numpy as np
import tensorflow as tf

_INITIALIZERS = {'glorot_uniform': tf.glorot_uniform_initializer(),
                 'glorot_normal': tf.glorot_normal_initializer(),
                 'orthogonal': tf.orthogonal_initializer(),
                 'lecun_normal': tf.initializers.lecun_normal(),
                 'lecum_uniform': tf.initializers.lecun_uniform(),
                 'he_normal': tf.initializers.he_normal(),
                 'he_uniform': tf.initializers.he_uniform(),
                 'constant_one': tf.constant_initializer(value=1.0),
                 'constant_zero': tf.constant_initializer(value=0.0),
                 }

_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
_DEFAULT_INIT = 'he_uniform'

def Linear(name, x, output_dim, bias=True, initializer=_DEFAULT_INIT):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        input_dim = shape[1]

        weights = tf.get_variable('Filters',
                shape=[input_dim, output_dim], 
                initializer=_INITIALIZERS[initializer],
                trainable=True)

        biases = tf.get_variable('Biases',
                shape=[output_dim],
                initializer=_BIAS_INITIALIZER,
                trainable=True)

        if bias:
            out = tf.add(tf.matmul(x, weights, name='Matmul'), biases, name='Output')
        else:
            out = tf.matmul(x, weights, name='Matmul')

    return out

def Conv2D(name, x, output_dim, kernel_size=3, stride=1, padding='SAME', bias=True, initializer=_DEFAULT_INIT):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        input_dim = shape[1]

        filters = tf.get_variable('Filters',
                shape=[kernel_size, kernel_size, input_dim, output_dim], 
                initializer=_INITIALIZERS[initializer],
                trainable=True)

        biases = tf.get_variable('Biases',
                shape=[output_dim],
                initializer=_BIAS_INITIALIZER,
                trainable=True)

        strides = [1,1,stride, stride]
        out = tf.nn.conv2d(x, filters, strides, padding, name='Conv2d', data_format='NCHW')

        if bias:
            out = tf.nn.bias_add(out, biases, data_format='NCHW')

        return out

def Normalize(name, x, method='BN', bn_is_training=True):
    if method == 'BN':
        return tf.layers.batch_normalization(x, training=bn_is_training, axis=1, 
                name=name+'.BN', reuse=tf.AUTO_REUSE, fused=True, momentum=0.9, epsilon=1e-5)

    elif method == 'LN':
        num_channels = x.get_shape().as_list()[1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            gamma = tf.get_variable('LN.Gamma', shape=[num_channels], initializer=tf.ones_initializer(), trainable=True)
            beta = tf.get_variable('LN.Beta', shape=[num_channels], initializer=tf.zeros_initializer(), trainable=True)

        gamma = tf.expand_dims(tf.expand_dims(gamma, -1), -1)
        beta = tf.expand_dims(tf.expand_dims(beta, -1), -1)

        mean, variance = tf.nn.moments(x, [1,2,3], keep_dims=True)
        output = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=1e-12)

        return output

    elif method is None:
        return x

def ResidualLayer(name, x, output_dim, kernel_size=3, stride=1, norm='BN', is_training=None, dropout=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        input_dim = shape[1]

        if input_dim == output_dim and stride == 1: shortcut = x
        else: shortcut = Conv2D(name+'.shortcut', x, output_dim, 1, stride, bias=False, initializer='glorot_uniform')

        output = Normalize(name+'.1', x, norm, is_training)
        output = tf.nn.relu(output)
        output = Conv2D(name+'.1', output, output_dim, kernel_size, stride, bias=False)

        output = Normalize(name+'.2', output, norm, is_training)
        output = tf.nn.relu(output)
        if dropout is not None:
            output = tf.layers.dropout(output, rate=dropout, training=is_training)
        output = Conv2D(name+'.2', output, output_dim, kernel_size, bias=False)

        return shortcut + output
