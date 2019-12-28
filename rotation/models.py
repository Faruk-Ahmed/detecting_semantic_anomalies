import os, sys
sys.path.append(os.getcwd())

from layers import Linear
from layers import Conv2D
from layers import Normalize
from layers import ResidualLayer

import numpy as np
import tensorflow as tf

from functools import partial

################################################# CIFAR ##########################################
def cifar10_base(images, DIM, is_training=None, dropout=None):
    output = Conv2D('classifier.conv.Init', images, DIM, initializer='glorot_uniform')

    output = ResidualLayer('classifier.conv.1.1', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.1.2', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.1.3', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.1.4', output, 2*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.2.1', output, 2*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.2.2', output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.2.3', output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.2.4', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.3.1', output, 4*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.3.2', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.3.3', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.3.4', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = Normalize('classifier.convout.3.NORM', output, bn_is_training=is_training)
    output = tf.nn.relu(output)

    return output

def cifar10_classifier(images, DIM, OUTPUT_DIM=10, is_training=None, dropout=None):
    output = cifar10_base(images, DIM, is_training, dropout)
    output = tf.reduce_mean(output, [2,3])
    output = Linear('classifier.class.final', output, OUTPUT_DIM, initializer='glorot_uniform')
    return output

def cifar10_rotation(images, DIM, OUTPUT_DIM=4, is_training=None, dropout=None):
    output = cifar10_base(images, DIM, is_training, dropout)
    output = tf.reduce_mean(output, [2,3])
    output = Linear('classifier.rot.final', output, OUTPUT_DIM, initializer='glorot_uniform')

    return output

################################################# STL-10 ##########################################
def stl10_base(images, DIM, is_training=None, dropout=None):
    output = Conv2D('classifier.conv.Init', images, DIM, initializer='glorot_uniform')

    output = ResidualLayer('classifier.conv.1.1', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.1.2', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.1.3', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.1.4', output, 2*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.2.1', output, 2*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.2.2', output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.2.3', output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.2.4', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.3.1', output, 4*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.3.2', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.3.3', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.3.4', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.4.1', output, 4*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.4.2', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.4.3', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.4.4', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = Normalize('classifier.convout.4.NORM', output, bn_is_training=is_training)
    output = tf.nn.relu(output)

    return output

def stl10_classifier(images, DIM, OUTPUT_DIM=10, is_training=None, dropout=None):
    output = stl10_base(images, DIM, is_training, dropout)
    output = tf.reduce_mean(output, [2,3])
    output = Linear('classifier.class.final', output, OUTPUT_DIM, initializer='glorot_uniform')
    return output

def stl10_rotation(images, DIM, OUTPUT_DIM=4, is_training=None, dropout=None):
    output = stl10_base(images, DIM, is_training, dropout)
    output = tf.reduce_mean(output, [2,3])
    output = Linear('classifier.rot.final', output, OUTPUT_DIM, initializer='glorot_uniform')
    return output

################################################# SubImagenets ##########################################
def subimagenet_base(images, DIM, is_training=None, dropout=None):
    output = Conv2D('classifier.conv.Init', images, 64, kernel_size=7, bias=False, initializer='glorot_uniform')
    output = tf.nn.max_pool(output, ksize=[1,1,3,3], strides=[1,1,2,2], padding='SAME', data_format='NCHW')

    output = ResidualLayer('classifier.conv.1', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.2', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.3', output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.4', output, 2*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.5', output, 2*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.6', output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.7', output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.8', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.9', output, 4*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.10', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.11', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.12', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('classifier.conv.13', output, 4*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.14', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.15', output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('classifier.conv.16', output, 4*DIM, is_training=is_training, dropout=dropout)

    output = Normalize('classifier.convout.NORM', output, bn_is_training=is_training)
    output = tf.nn.relu(output)

    return output

def subimagenet_classifier(images, DIM, OUTPUT_DIM, is_training=None, dropout=None):
    output = subimagenet_base(images, DIM, is_training, dropout)
    output = tf.reduce_mean(output, [2,3])
    output = Linear('classifier.class.final', output, OUTPUT_DIM, initializer='glorot_uniform')
    return output

def subimagenet_rotation(images, DIM, OUTPUT_DIM=4, is_training=None, dropout=None):
    output = subimagenet_base(images, DIM, is_training, dropout)
    output = tf.reduce_mean(output, [2,3])
    output = Linear('classifier.rot.final', output, OUTPUT_DIM, initializer='glorot_uniform')
    return output
