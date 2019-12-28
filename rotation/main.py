import os, sys, time, locale
sys.path.append(os.getcwd())
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from models import cifar10_classifier, stl10_classifier
from models import cifar10_rotation, stl10_rotation

import dataloaders.cifar10_loo, dataloaders.stl10_loo

import numpy as np
import sklearn.metrics

import time
import random
import functools
import cPickle as pkl

import argparse

parser = argparse.ArgumentParser()
locale.setlocale(locale.LC_ALL, '')

######################################## Args ######################################## 
parser.add_argument('-c', '--anomaly', help='Amonaly', type=int, default=0)
parser.add_argument('-dc', '--dim_class', help='Classifier dimensions', default=160, type=int)
parser.add_argument('-bs', '--batch_size', help='Batch size', default=128, type=int)
parser.add_argument('-lr', '--learning_rate', help='Initial learning rate', type=float, default=1e-1)
parser.add_argument('-d', '--dataset', help='Which dataset?', required=True, type=str, default='cifar10', choices=['cifar10', 'stl10'])
parser.add_argument('-r', '--rotate', help='Rotate?', action='store_true')
parser.add_argument('-ngpu', '--num_gpus', help='How many GPUs?', default=4, type=int)
parser.add_argument('-dr', '--dropout', help='Dropout?', type=float, default=0.3)
parser.add_argument('-wc', '--weight_cost', type=float, default=0.5)
parser.add_argument('-v', '--validate', help='Use validation set', action='store_true')

args = vars(parser.parse_args())

ANOMALY = args['anomaly']
DIM_CLASS = args['dim_class']
LR = args['learning_rate']
DATASET = args['dataset']
ROTATE = args['rotate']
NUM_GPUS = args['num_gpus']
BATCH_SIZE = args['batch_size']*NUM_GPUS
DROPOUT = args['dropout']
WC = args['weight_cost']
VALIDATION = args['validate']

local_vars = [(k,v) for (k,v) in locals().items() if (k.isupper())]
for var_name, var_value in local_vars:
    print "\t{}: {}".format(var_name, var_value)

######################################## Data loaders and stuff ######################################## 
fmap = {'cifar10': {'loader': dataloaders.cifar10_loo.load,
                    'classifier': cifar10_classifier,
                    'rotationer': cifar10_rotation,
                    'total_size': 50000, 
                    'dims': [3, 32, 32]},
             'stl10': {'loader': dataloaders.stl10_loo.load,
                       'classifier': stl10_classifier,
                       'rotationer': stl10_rotation,
                       'total_size': 10000, 
                       'dims': [3, 96, 96]},
             }[DATASET]

train_data, dev_data, _ = fmap['loader'](train_batch_size=BATCH_SIZE,
                                         test_batch_size=50,
                                         remove_class=ANOMALY,
                                         validation=VALIDATION,
                                         data_dir=os.environ['DATA_DIR'])

if ROTATE:
    rot_train_data, rot_dev_data, _ = fmap['loader'](train_batch_size=BATCH_SIZE//4,
                                                     test_batch_size=BATCH_SIZE//4,
                                                     remove_class=ANOMALY,
                                                     validation=VALIDATION,
                                                     data_dir=os.environ['DATA_DIR'])

iters_per_epoch = fmap['total_size']/BATCH_SIZE
TIMES = {'print_every': 100, 'test_every': 500}

FIRST_CUT = 60*iters_per_epoch
SECOND_CUT = 120*iters_per_epoch
THIRD_CUT = 160*iters_per_epoch

TOTAL_ITERS = iters_per_epoch*200
TIMES['stop_after'] = TOTAL_ITERS

##################################################################################
C, H, W = fmap['dims']
def preprocess_images(images):
    images = tf.transpose(images, [0,2,3,1])
    images = tf.pad(images, [[0,0], [H//8,H//8], [W//8,W//8], [0,0]], mode='REFLECT')
    images = tf.map_fn(lambda image: tf.image.random_crop(image, [H,W,C]), images)
    images = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), images)
    images = tf.transpose(images, [0,3,1,2])
    return images

def rot_preprocess(data):
    N = data.shape[0]

    images0 = data
    images1 = data[:, :, ::-1, :]
    images2 = np.transpose(data, [0, 1, 3, 2])
    images3 = np.transpose(data[:, :, ::-1, :], [0, 1, 3, 2])

    all_images = np.concatenate((images0, images1, images2, images3), axis=0)
    all_rotation_labels = np.concatenate((np.zeros(N), np.ones(N), 2*np.ones(N), 3*np.ones(N)), axis=0).astype('int32')

    # This shuffle is for multi-GPU training:
    rng_state = np.random.get_state()
    np.random.shuffle(all_images)
    np.random.set_state(rng_state)
    np.random.shuffle(all_rotation_labels)
                                     
    return all_images, all_rotation_labels

##################################################################################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    # Tensor Placeholders:
    all_images = tf.placeholder(tf.int32, shape=[None] + fmap['dims'], name='all_images')
    all_labels = tf.placeholder(tf.int32, shape=[None], name='all_labels')
    all_rotations = tf.placeholder(tf.int32, shape=[None], name='all_rotations')

    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')
    rotate = tf.placeholder(tf.bool, shape=None, name='rotate')
    lr = tf.placeholder(tf.float32, shape=None, name='lr')

    dataset_mean = tf.placeholder(tf.float32, shape=[1,3,1,1], name='data_mean')
    dataset_std = tf.placeholder(tf.float32, shape=[1,3,1,1], name='data_std')

    ####################################################################################
    split_images = tf.split(all_images, NUM_GPUS, axis=0)
    split_labels = tf.split(all_labels, NUM_GPUS, axis=0)
    split_rotations = tf.split(all_rotations, NUM_GPUS, axis=0)

    tower_class_costs, tower_class_accuracies = [], []
    tower_rotation_costs, tower_rotation_accuracies = [], []

    for device_index, (tower_images, tower_labels, tower_rotations) in enumerate(zip(split_images, split_labels, split_rotations)):
        with tf.device(tf.DeviceSpec(device_type='GPU', device_index=device_index)):
            scaled_images = (tf.cast(tower_images, 'float32') - dataset_mean)/dataset_std

            # Data-augmentation for the auxiliary task underperformed in preliminary experiments
            scaled_images = tf.cond(rotate, 
                    lambda: scaled_images, 
                    lambda: tf.cond(is_training, lambda: preprocess_images(scaled_images), lambda: scaled_images))

            classifier_output = fmap['classifier'](scaled_images, DIM_CLASS, 9, is_training, dropout=DROPOUT)

            tower_class_xentropy_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tower_labels, logits=classifier_output))
            class_predictions = tf.cast(tf.argmax(classifier_output, axis=1), 'int32')
            tower_class_accuracy = tf.contrib.metrics.accuracy(labels=tower_labels, predictions=class_predictions)

            tower_class_costs.append(tower_class_xentropy_cost)
            tower_class_accuracies.append(tower_class_accuracy)

            if ROTATE:
                rotation_output = fmap['rotationer'](scaled_images, DIM_CLASS, 4, is_training, dropout=DROPOUT)

                tower_rot_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tower_rotations, logits=rotation_output))
                rot_predictions = tf.cast(tf.argmax(rotation_output, axis=1), 'int32')
                tower_rot_acc = tf.contrib.metrics.accuracy(labels=tower_rotations, predictions=rot_predictions)

                tower_rotation_costs.append(tower_rot_cost)
                tower_rotation_accuracies.append(tower_rot_acc)

    class_xentropy_cost = tf.reduce_mean([tf.concat(x, axis=0) for x in tower_class_costs], axis=0)
    class_accuracy = tf.reduce_mean([tf.concat(x, axis=0) for x in tower_class_accuracies], axis=0)

    class_train_vars = [param for param in tf.trainable_variables() if 'classifier' in param.name and 'rot.final' not in param.name]
    class_L2_regularization = tf.add_n([tf.nn.l2_loss(v) for v in class_train_vars]) * 5e-4

    class_optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    classifier_gv = class_optimizer.compute_gradients(class_xentropy_cost+class_L2_regularization, 
            var_list=class_train_vars, colocate_gradients_with_ops=True)

    if ROTATE:
        rotation_xentropy_cost = tf.reduce_mean([tf.concat(x, axis=0) for x in tower_rotation_costs], axis=0)
        rotation_accuracy = tf.reduce_mean([tf.concat(x, axis=0) for x in tower_rotation_accuracies], axis=0)

        rot_train_vars = [param for param in tf.trainable_variables() if 'classifier' in param.name and 'class.final' not in param.name]
        rot_L2_regularization = tf.add_n([tf.nn.l2_loss(v) for v in rot_train_vars]) * 5e-4

        rot_optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        rotation_gv = rot_optimizer.compute_gradients(WC*rotation_xentropy_cost+rot_L2_regularization, 
                var_list=rot_train_vars, colocate_gradients_with_ops=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        class_train_op = class_optimizer.apply_gradients(classifier_gv)
        if ROTATE: rot_train_op = rot_optimizer.apply_gradients(rotation_gv)


    ######################################## Anomaly detection  ########################################
    _, in_dev_data, out_dev_data  = fmap['loader'](train_batch_size=BATCH_SIZE,
                                                   test_batch_size=50,
                                                   remove_class=ANOMALY,
                                                   data_dir=os.environ['DATA_DIR'])
    msp_sfx = tf.nn.softmax(classifier_output)
    msp_min_dist = tf.reduce_max(msp_sfx, axis=1)

    ODIN_EPSILON = 5e-5
    ODIN_TEMPERATURE = 1000
    
    odin_xentropy_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=class_predictions,
        logits=classifier_output/ODIN_TEMPERATURE))
    gradient = tf.gradients(odin_xentropy_cost, [scaled_images])[0]

    gradient = tf.cast(tf.math.greater_equal(gradient, 0.0), 'float32')
    gradient = 2.0*(gradient - 0.5)/dataset_std

    preprocessed_image = scaled_images - ODIN_EPSILON*gradient
    odin_classifier_output = fmap['classifier'](preprocessed_image, DIM_CLASS, 9, is_training, dropout=DROPOUT)
    odin_sfx = tf.nn.softmax(odin_classifier_output/ODIN_TEMPERATURE)
    odin_min_dist = tf.reduce_max(odin_sfx, axis=1)

    def signal(_images):
        _msp_signal, _odin_signal = session.run([msp_min_dist, odin_min_dist], 
                feed_dict={tower_images: _images, dataset_mean: _mean, dataset_std: _std, is_training: False, rotate: False})
        return -_msp_signal, -_odin_signal

    def compute_av_prec():
        msp_ins, msp_outs, odin_ins, odin_outs = [], [], [], []
        in_test_gen, out_test_gen = in_dev_data(), out_dev_data()

        while True:
            try: _data = in_test_gen.next()
            except StopIteration: break

            msp_scores, odin_scores = signal(_data[0])
            msp_ins += [msp_scores]
            odin_ins += [odin_scores]

        while True:
            try: _data = out_test_gen.next()
            except StopIteration: break

            msp_scores, odin_scores = signal(_data[0])
            msp_outs += [msp_scores]
            odin_outs += [odin_scores]

        msp_ins, msp_outs = np.concatenate(msp_ins), np.concatenate(msp_outs)
        odin_ins, odin_outs = np.concatenate(odin_ins), np.concatenate(odin_outs)

        TRUE_LABELS = np.hstack((np.zeros(len(msp_ins)), np.ones(len(msp_outs))))
        MSP_TESTS = np.hstack((msp_ins, msp_outs))
        ODIN_TESTS = np.hstack((odin_ins, odin_outs))

        msp_av_prec = 100.*sklearn.metrics.average_precision_score(TRUE_LABELS, MSP_TESTS)
        odin_av_prec = 100.*sklearn.metrics.average_precision_score(TRUE_LABELS, ODIN_TESTS)

        return msp_av_prec, odin_av_prec

    ######################################## Pre-training stuff ######################################## 
    train_gen = train_data()
    all_data = []
    while True:
        try: _data = train_gen.next()
        except StopIteration: break
        all_data += [_data[0]]
    all_data = np.concatenate(all_data)
    _mean, _std = np.mean(all_data, (0,2,3), keepdims=True), np.std(all_data, (0,2,3), keepdims=True)

    running_avg_cost, running_avg_accuracy = 0.0, 0.0
    tr_costs, dev_costs, tr_accuracies, dev_accuracies = [], [], [], []
    av_precs = []

    if ROTATE:
        running_avg_rot_cost, running_avg_rot_accuracy = 0.0, 0.0
        rot_tr_costs, rot_dev_costs, rot_tr_accuracies, rot_dev_accuracies = [], [], [], []

    train_gen = train_data()
    if ROTATE: rot_train_gen = rot_train_data()

    _lr = LR
    old_epoch = 0

    ######################################## Training ######################################## 
    session.run(tf.initialize_all_variables())
    for iteration in xrange(TIMES['stop_after']):
        epoch = iteration/iters_per_epoch
        if iteration > TIMES['stop_after']: break

        if ROTATE:
            try: _data = rot_train_gen.next()
            except StopIteration:
                rot_train_gen = rot_train_data()
                _data = rot_train_gen.next()

            _images, _rotations = rot_preprocess(_data[0])
            _rotation_cost, _rotation_accuracy,  _ = session.run([rotation_xentropy_cost, rotation_accuracy, rot_train_op],
                    feed_dict={all_images: _images, dataset_mean: _mean, dataset_std: _std, 
                        all_rotations: _rotations, lr: _lr, is_training: True, rotate: True}) 

        try: _data = train_gen.next()
        except StopIteration:
            train_gen = train_data()
            _data = train_gen.next()

        _classifier_cost, _classification_accuracy, _ = session.run([class_xentropy_cost, class_accuracy, class_train_op],
                feed_dict={all_images: _data[0], dataset_mean: _mean, dataset_std: _std, all_labels: _data[1], lr: _lr, is_training: True, rotate: False}) 

        running_avg_cost += (_classifier_cost - running_avg_cost)/(iteration+1)
        running_avg_accuracy += (_classification_accuracy - running_avg_accuracy)/(iteration+1)

        if ROTATE:
            running_avg_rot_cost += (_rotation_cost - running_avg_rot_cost)/(iteration+1)
            running_avg_rot_accuracy += (_rotation_accuracy - running_avg_rot_accuracy)/(iteration+1)

        if iteration % TIMES['print_every'] == 0:
            print 'iter', iteration, 'lr', _lr, 'classifier cost =', running_avg_cost, 'accuracy =', running_avg_accuracy,
            if ROTATE: print 'rotation cost =', running_avg_rot_cost, 'rotation accuracy =', running_avg_rot_accuracy
            else: print ''

        if iteration <= FIRST_CUT: _lr = LR
        elif iteration > FIRST_CUT and iteration <= SECOND_CUT: _lr = LR*0.2
        elif iteration > SECOND_CUT and iteration <= THIRD_CUT: _lr = LR*0.2*0.2
        elif iteration > THIRD_CUT: _lr = LR*0.2*0.2*0.2

        if iteration % TIMES['test_every'] == 0:
            _dev_cost, _dev_acc = 0.0, 0.0
            if ROTATE: _dev_rot_cost, _dev_rot_acc = 0.0, 0.0

            dev_gen = dev_data()
            _dev_iter = 0
            while True:
                try: _data = dev_gen.next()
                except StopIteration: break

                if ROTATE:
                    _images, _rotations = rot_preprocess(_data[0])[:BATCH_SIZE]
                    _rot_cost, _rot_acc = session.run([rotation_xentropy_cost, rotation_accuracy], 
                            feed_dict={all_images: _images, dataset_mean: _mean, dataset_std: _std,
                                       all_rotations: _rotations, is_training: False, rotate: True})

                    _dev_rot_cost += (_rot_cost - _dev_rot_cost)/(_dev_iter+1)
                    _dev_rot_acc += (_rot_acc - _dev_rot_acc)/(_dev_iter+1)

                _cost, _acc = session.run([tower_class_xentropy_cost, tower_class_accuracy], 
                        feed_dict={tower_images: _data[0], dataset_mean: _mean, dataset_std: _std,
                                   tower_labels: _data[1], is_training: False, rotate: False})

                _dev_cost += (_cost - _dev_cost)/(_dev_iter+1)
                _dev_acc += (_acc - _dev_acc)/(_dev_iter+1)
                _dev_iter += 1

            print ' '*20, '--  dev cost =', _dev_cost, ', dev acc. rate =', _dev_acc, 
            if ROTATE: print 'dev rot cost =', _dev_rot_cost, 'dev rot acc =', _dev_rot_acc,
            else: print '',
            print 'Anomaly detection:', compute_av_prec()

            tr_costs += [running_avg_cost]
            dev_costs += [_dev_cost]
            tr_accuracies += [running_avg_accuracy]
            dev_accuracies += [_dev_acc]
            if ROTATE:
                rot_tr_accuracies += [running_avg_rot_accuracy]
                rot_tr_costs += [running_avg_rot_cost]
                rot_dev_costs += [_dev_rot_cost]
                rot_dev_accuracies += [_dev_rot_acc]

        if epoch != old_epoch and epoch > 90:
            old_epoch = epoch
            av_precs += [compute_av_prec()]

        if iteration == TIMES['stop_after']-1:
            av_precs += [compute_av_prec()]

# The average precision keeps wiggling around till the end, so average over the last 10 epochs:
print 'Final: Classification accuracy = {}, Average precision (MSP, ODIN) = {}'.format(_dev_acc, np.mean(av_precs[-10:], 0))
