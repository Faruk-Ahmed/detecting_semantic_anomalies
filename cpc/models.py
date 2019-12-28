import os, sys
sys.path.append(os.getcwd())

from layers import Linear
from layers import Conv2D
from layers import Normalize
from layers import ResidualLayer
from layers import RowMaskResidualLayer

from math import ceil
import numpy as np
import tensorflow as tf

from functools import partial

DIM = 64
CPC_NORM = 'CBN'
HDIM = 4*DIM

def same_padding(in_height, in_width, filter_height, stride):
    pad_total = filter_height - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    return [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]

def shared_feature_extractor(patches, selfsup, is_training=None):
    output = Conv2D('classifier.featureextractor.Conv.Init', patches, 64, kernel_size=7, stride=2, padding='SAME', bias=False, initializer='glorot_uniform')

    _shape = output.get_shape().as_list()
    output = tf.pad(output, same_padding(_shape[2], _shape[3], 3, 2), 'SYMMETRIC')
    output = tf.nn.max_pool(output, ksize=[1,1,3,3], strides=[1,1,2,2], padding='VALID', data_format='NCHW')

    output = ResidualLayer('classifier.conv.1', output, DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.2', output, DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.3', output, DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.4', output, 2*DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)

    output = ResidualLayer('classifier.conv.5', output, 2*DIM, norm=CPC_NORM, stride=2, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.6', output, 2*DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.7', output, 2*DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.8', output, 4*DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)

    output = ResidualLayer('classifier.conv.9', output, 4*DIM, norm=CPC_NORM, stride=2, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.10', output, 4*DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.11', output, 4*DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)
    output = ResidualLayer('classifier.conv.12', output, 4*DIM, norm=CPC_NORM, is_training=is_training, labels=selfsup)

    return output

def feature_extractor(patches, selfsup, is_training=None):
    output = shared_feature_extractor(patches, selfsup, is_training=is_training)
    return output

CPC_ResLayer = partial(ResidualLayer, norm=None)
def context_network(spatial_features, cdim):
    output = spatial_features
    output = Conv2D('context.Conv.init', output, cdim, kernel_size=1)
    
    output = RowMaskResidualLayer('context.MaskRes.1', output, cdim, kernel_size=3)

    output = CPC_ResLayer('context.Res.1', output, cdim, kernel_size=1)
    output = CPC_ResLayer('context.Res.2', output, cdim, kernel_size=1)
    output = CPC_ResLayer('context.Res.3', output, cdim, kernel_size=1)
    output = CPC_ResLayer('context.Res.4', output, cdim, kernel_size=1)
    output = tf.nn.relu(output)

    output = Conv2D('context.Conv.out', output, cdim, kernel_size=1)
    return output

GRID_SIZE = 6
def predict_features(context_predictions, cdim):
    bs = tf.shape(context_predictions)[0]
    output = tf.transpose(context_predictions, [0,3,1,2])
    output = tf.reshape(output, [bs*GRID_SIZE, cdim, GRID_SIZE])

    all_predictions = []
    for row in range(1,GRID_SIZE):
        predictions = []

        for previous_row in range(0,row):
            if (row-previous_row) > 5: pass
            else:
                with tf.variable_scope('featurepredictor.stepsback{}'.format(row-previous_row), reuse=tf.AUTO_REUSE):
                    W = tf.get_variable('Filter', shape=[cdim, HDIM], initializer=tf.initializers.glorot_uniform(), trainable=True)
                row_prediction = tf.matmul(output[:, :, previous_row], W)
                predictions += [row_prediction]

        predictions = tf.stack(predictions, axis=2)
        predictions = tf.reshape(predictions, [bs, GRID_SIZE, HDIM, -1])
        predictions = tf.transpose(predictions, [0,2,3,1])

        all_predictions += [predictions]

    return all_predictions

def cpc_classifier(scaled_images, OUTPUT_DIM, selfsup, is_training=None):
    output = shared_feature_extractor(scaled_images, selfsup, is_training)

    output = ResidualLayer('classifier.conv.13', output, 4*DIM, stride=2, is_training=is_training)
    output = ResidualLayer('classifier.conv.14', output, 4*DIM, is_training=is_training)
    output = ResidualLayer('classifier.conv.15', output, 4*DIM, is_training=is_training)
    output = ResidualLayer('classifier.conv.16', output, 4*DIM, is_training=is_training)
    
    output = Normalize('classifier.final', output, 'BN', bn_is_training=is_training)
    output = tf.nn.relu(output)
    output = tf.reduce_mean(output, [2,3])

    output = Linear('classifier.final', output, OUTPUT_DIM, 'glorot_uniform')
    return output
