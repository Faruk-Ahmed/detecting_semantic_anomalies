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

def same_padding(in_height, in_width, filter_height, stride):
    pad_total = filter_height - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    return [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]

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

        if padding == 'SAME':
            _shape = x.get_shape().as_list()
            out = tf.pad(x, same_padding(_shape[2], _shape[3], kernel_size, stride), 'SYMMETRIC')
        out = tf.nn.conv2d(out, filters, strides, 'VALID', name='Conv2d', data_format='NCHW')

        if bias:
            out = tf.nn.bias_add(out, biases, data_format='NCHW')

        return out

def Normalize(name, x, method='BN', bn_is_training=True, labels=None):
    if method == 'BN':
        num_channels = x.get_shape().as_list()[1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            gamma = tf.get_variable('BN.Gamma', 
                    shape=[num_channels], 
                    initializer=tf.ones_initializer(), 
                    trainable=True)
            beta = tf.get_variable('BN.Beta', 
                    shape=[num_channels], 
                    initializer=tf.zeros_initializer(), 
                    trainable=True)

            moving_mean = tf.get_variable('BN.Moving_mean', 
                    shape=[num_channels], 
                    trainable=False, 
                    initializer=tf.zeros_initializer())
            moving_variance = tf.get_variable('BN.Moving_variance', 
                    shape=[num_channels], 
                    trainable=False, 
                    initializer=tf.ones_initializer())

        def _bn_training():
 	     training_bn_output, mean, variance = tf.nn.fused_batch_norm(x, 
                     scale=gamma, 
                     offset=beta, 
                     epsilon=1e-5, 
                     data_format='NCHW')

             momentum = 0.9
             update_moving_mean = (1.-momentum)*moving_mean + momentum*mean
             update_moving_variance = (1.-momentum)*moving_variance + momentum*variance
             
             update_ops = [moving_mean.assign(update_moving_mean), moving_variance.assign(update_moving_variance)]
             with tf.control_dependencies(update_ops):
                 return tf.identity(training_bn_output)
                 
        def _bn_inference():
            inference_bn_output, _, _ = tf.nn.fused_batch_norm(x,
                    mean=moving_mean, 
                    variance=moving_variance,
                    scale=gamma, 
                    offset=beta,
                    is_training=False,
                    epsilon=1e-5, 
                    data_format='NCHW')
            return inference_bn_output

        output = tf.cond(bn_is_training, lambda: _bn_training(), lambda: _bn_inference())
        return output

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

    elif method == 'CBN':
        if labels is None: raise Exception('This requires labels (task id) to be provided')
        num_labels = 2   # Sorry for the hard-code :(
        num_channels = x.get_shape().as_list()[1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            label_gamma = tf.get_variable('BN.Gamma', 
                    shape=[num_labels,num_channels], 
                    initializer=tf.ones_initializer(), 
                    trainable=True)
            label_beta = tf.get_variable('BN.Beta', 
                    shape=[num_labels,num_channels], 
                    initializer=tf.zeros_initializer(), 
                    trainable=True)

            label_moving_mean = tf.get_variable('BN.Moving_mean', 
                    shape=[num_labels,num_channels], 
                    trainable=False, 
                    initializer=tf.zeros_initializer())
            label_moving_variance = tf.get_variable('BN.Moving_variance', 
                    shape=[num_labels,num_channels], 
                    trainable=False, 
                    initializer=tf.ones_initializer())

        # labels is actually a 'label', so this is good (replace with tf.gather for separate labels per image)
        gamma = label_gamma[labels,:]
        beta = label_beta[labels,:]
        moving_mean = label_moving_mean[labels,:]
        moving_variance = label_moving_variance[labels,:]

        def _bn_training():
 	     training_bn_output, mean, variance = tf.nn.fused_batch_norm(x, 
                     scale=gamma, 
                     offset=beta, 
                     epsilon=1e-5, 
                     data_format='NCHW')

             momentum = tf.cond(tf.equal(labels, 0), lambda: tf.constant([[0.9],[0.0]]), lambda: tf.constant([[0.0],[0.9]]))
             update_moving_mean = (1.-momentum)*label_moving_mean + momentum*mean
             update_moving_variance = (1.-momentum)*label_moving_variance + momentum*variance
             
             update_ops = [label_moving_mean.assign(update_moving_mean), label_moving_variance.assign(update_moving_variance)]
             with tf.control_dependencies(update_ops):
                 return tf.identity(training_bn_output)
                 
        def _bn_inference():
            inference_bn_output, _, _ = tf.nn.fused_batch_norm(x,
                    mean=moving_mean, 
                    variance=moving_variance,
                    scale=gamma, 
                    offset=beta,
                    is_training=False,
                    epsilon=1e-5, 
                    data_format='NCHW')
            return inference_bn_output

        output = tf.cond(bn_is_training, lambda: _bn_training(), lambda: _bn_inference())
        return output

    elif method == 'CLN':
        if labels is None: raise Exception('This requires labels (task id) to be provided')
        num_labels = 2   # Sorry for the hard-code :(

        x = tf.convert_to_tensor(x)
        x_shape = x.get_shape()
        num_channels = x.get_shape().as_list()[1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            gamma = tf.get_variable('CLN.Gamma', shape=[num_labels, num_channels], initializer=tf.ones_initializer(), trainable=True)
            beta = tf.get_variable('CLN.Beta', shape=[num_labels, num_channels], initializer=tf.zeros_initializer(), trainable=True)

        # labels is actually a 'label', so this is good (replace with tf.gather for separate labels per image)
        label_gamma = tf.expand_dims(tf.expand_dims(gamma[labels, :], -1), -1)
        label_beta = tf.expand_dims(tf.expand_dims(beta[labels, :] , -1), -1)

        mean, variance = tf.nn.moments(x, [1,2,3], keep_dims=True)
        output = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=label_beta, scale=label_gamma, variance_epsilon=1e-12)

        return output

    elif method is None: return x

def ResidualLayer(name, x, output_dim, kernel_size=3, stride=1, norm='BN', is_training=None, dropout=None, labels=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        input_dim = shape[1]

        if input_dim == output_dim and stride == 1: shortcut = x
        else: shortcut = Conv2D(name+'.shortcut', x, output_dim, 1, stride, bias=False, initializer='glorot_uniform')

        output = Normalize(name+'.1', x, norm, is_training, labels=labels)
        output = tf.nn.relu(output)
        output = Conv2D(name+'.1', output, output_dim, kernel_size, stride, bias=False)

        output = Normalize(name+'.2', output, norm, is_training, labels=labels)
        output = tf.nn.relu(output)

        if dropout is not None:
            output = tf.layers.dropout(output, rate=dropout, training=is_training)
        output = Conv2D(name+'.2', output, output_dim, kernel_size, bias=False)

        return shortcut + output

def RowMaskConv2D(name, x, output_dim, kernel_size=5, padding='SAME', bias=True, initializer=_DEFAULT_INIT):
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

        mask = np.zeros((kernel_size, kernel_size))
        mask[:kernel_size//2+1, :] = 1.
        filters = filters*mask[:,:,None,None]

        strides = [1,1,1,1]
        out = tf.nn.conv2d(x, filters, strides, padding, name='Conv2d', data_format='NCHW')

        if bias:
            out = tf.nn.bias_add(out, biases, data_format='NCHW')

        return out

def RowMaskResidualLayer(name, x, output_dim, kernel_size=3, norm=None, is_training=None, nonlinearity=tf.nn.relu):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        input_dim = shape[1]

        if input_dim == output_dim:
            shortcut = x
        else:
            shortcut = Conv2D(name+'.shortcut', x, output_dim, 1, stride, bias=False, initializer='glorot_uniform')

        output = Normalize(name+'.1', x, norm, is_training)
        output = nonlinearity(output)
        output = RowMaskConv2D(name+'.1', output, output_dim, kernel_size, bias=False)

        output = Normalize(name+'.2', output, norm, is_training)
        output = nonlinearity(output)

        if norm is None: output = RowMaskConv2D(name+'.2', output, output_dim, 1, bias=True)
        else: output = RowMaskConv2D(name+'.2', output, output_dim, 1, bias=False)

        return shortcut + output
