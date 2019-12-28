"""
   There's a lot of hard-coding and inelegance around, sorry :(
   Also, you'd have to swap out the dataloading stuff with your own,
"""
import os, sys
sys.modules['theano'] = None
sys.path.append(os.getcwd())

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from models import cpc_classifier
from models import feature_extractor as fext
from models import context_network
from models import predict_features

from layers import Normalize
from models import CPC_NORM, HDIM, GRID_SIZE

import sklearn.metrics
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import time
import random
import functools
import cPickle as pkl 
import argparse
parser = argparse.ArgumentParser()

##################################################################################
parser.add_argument('-c', '--anomaly', help='Amonaly', type=int, default=0)
parser.add_argument('-bs', '--batch_size', help='Batch size', default=64, type=int)
parser.add_argument('-cd', '--context_dim', help="Context dimensionality", default=256, type=int)
parser.add_argument('-lr', '--learning_rate', help='Initial learning rate', type=float, default=1e-1)
parser.add_argument('-d', '--dataset', help="which dataset", required=True, type=str, 
        choices=['snake', 'car', 'dog', 'fungus', 'spider'])
parser.add_argument('-s', '--selfsup', help='CPC?', action='store_true')
parser.add_argument('-wc', '--weight_cost', type=float, default=10.0)
parser.add_argument('-v', '--validate', help='Use validation set', action='store_true')

args = vars(parser.parse_args())

ANOMALY = args['anomaly']
CONTEXT_DIM = args['context_dim']
LR = args['learning_rate']
DATASET = args['dataset']
SELFSUP = args['selfsup']
BATCH_SIZE = args['batch_size']
WC = args['weight_cost']
VALIDATION = args['validate']

CPC_BATCHSIZE = 32

local_vars = [(k,v) for (k,v) in locals().items() if (k.isupper())]
for var_name, var_value in local_vars:
    print "\t{}: {}".format(var_name, var_value)

TIMES = {
    'print_every': 200,
    'test_every': 1000,
}

DATA_DIR = os.path.join(os.environ['DATA_DIR'], 'subimagenets', DATASET)
labelmap = pkl.load(open(os.path.join(DATA_DIR, '{}_labelmap.pkl'.format(DATASET))))
OUTPUT_DIM = len(labelmap.keys())

##################################################################################
C, H, W = 3, 224, 224
def preprocess_images(images):
    images = tf.transpose(images, [0,2,3,1])
    images = tf.map_fn(lambda image: tf.image.random_crop(image, [H,W,C]), images)
    images = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), images)
    images = tf.transpose(images, [0,3,1,2])
    return images

def center_crop(images):
    images = images[:, :, 128-112:128+112, 128-112:128+112]
    return images

def leave_out_class(data, c, restore_size=True):
    images, labels = data[0], data[1]
    bs = images.shape[0]

    in_dist_pos = np.in1d(labels, np.setdiff1d(range(OUTPUT_DIM), c))
    images = images[in_dist_pos, ...]
    labels = labels[in_dist_pos, ...]

    if np.sum(np.where(labels == c, 1, 0)) > 0:
        raise Exception('wait what')

    sub = np.where(labels > c, 1, 0)
    labels = labels - sub

    if restore_size:
        new_size = images.shape[0]
        this_many_more = bs-new_size

        _replace = True if (this_many_more > new_size) else False
        indx = np.random.choice(range(new_size), this_many_more, replace=_replace)
        
        images = np.concatenate((images, images[indx][:,:,:,::-1]), axis=0)
        labels = np.concatenate((labels, labels[indx]), axis=0)

    return (images, labels)

##################################################################################
def feature_extractor(images, selfsup, is_training):
    features = fext(images, selfsup, is_training)
    features = Normalize('featureextractor.final.NORM', features, CPC_NORM, labels=selfsup, bn_is_training=is_training)
    features = tf.nn.relu(features)
    features = tf.reduce_mean(features, [2,3])
    return features

def mess_them_up_good(patches):
    patches = tf.transpose(patches, [0,2,3,1])
    
    # Randomly drop 2 of the color channels from patches:
    patches = tf.map_fn(lambda patch: patch[:,:,tf.random_uniform([1],0,3,dtype=tf.dtypes.int32)[0]][:,:,tf.newaxis], patches)
    patches = tf.tile(patches, [1,1,1,3])

    # Randomly flip patches:
    patches = tf.map_fn(lambda patch: tf.image.random_flip_left_right(patch), patches)
    
    # Spatial jitter to throw off continuation shortcuts:
    patches = tf.map_fn(lambda patch: tf.image.random_crop(patch, [56,56,3]), patches)
    patches = tf.pad(patches, [[0,0], [4,4], [4,4], [0,0]], mode='REFLECT')

    patches = tf.transpose(patches, [0,3,1,2])
    return patches

def extract(images, selfsup, is_training):
    bs = tf.shape(images)[0]
    patches = get_patches(images)
    patches = mess_them_up_good(patches)

    patch_features = feature_extractor(patches, selfsup, is_training)
    patch_features = tf.reshape(patch_features, [bs,GRID_SIZE,GRID_SIZE,HDIM])
    patch_features = tf.transpose(patch_features, [0,3,1,2])
    return patch_features

def shuffle(images, selfsup, is_training):
    same_images = tf.identity(images)

    patches_from_same_images = get_patches(same_images)
    patches_from_same_images = tf.reshape(patches_from_same_images, [-1,GRID_SIZE,GRID_SIZE,3,64,64])
    patches_from_same_images = tf.transpose(patches_from_same_images, [1,2,0,3,4,5])
    patches_from_same_images = tf.reshape(patches_from_same_images, [GRID_SIZE**2,-1,3,64,64])
    patches_from_same_images = tf.random_shuffle(patches_from_same_images)
    patches_from_same_images = tf.reshape(patches_from_same_images, [GRID_SIZE,GRID_SIZE,-1,3,64,64])
    patches_from_same_images = tf.transpose(patches_from_same_images, [2,0,1,3,4,5])
    patches_from_same_images = tf.reshape(patches_from_same_images, [-1,3,64,64])
    patches_from_same_images = patches_from_same_images

    negative_patches_same = tf.stop_gradient(mess_them_up_good(patches_from_same_images))

    same_features = feature_extractor(negative_patches_same, selfsup, is_training)
    same_features = tf.reshape(same_features, [-1,GRID_SIZE,GRID_SIZE,HDIM])
    same_features = tf.transpose(same_features, [0,3,1,2])

    return same_features

def get_patches(images):
    bs = tf.shape(images)[0]
    output = tf.transpose(images, [0,2,3,1])
    patches_of_images = tf.extract_image_patches(images=output,
         ksizes=[1,64,64,1],
         strides=[1,32,32,1],
         rates=[1,1,1,1],
         padding='VALID') 
    patches_of_images = tf.reshape(patches_of_images, [bs*GRID_SIZE*GRID_SIZE,64,64,3])
    patches_of_images = tf.transpose(patches_of_images, [0,3,1,2])
    return patches_of_images

def compute_scores(features, predictions, negative_features_same):
    positive_step_scores = [[] for i in xrange(GRID_SIZE-1)]
    negative_step_scores_same = [[] for i in xrange(GRID_SIZE-1)]

    for i, prediction in enumerate(predictions):
        tiled_feature = features[:,:,i+1,:][:,:,tf.newaxis,:]
        tiled_negative_feature_same = negative_features_same[:,:,i+1,:][:,:,tf.newaxis,:]

        positive_products = tf.reduce_mean(tf.multiply(tiled_feature, prediction), reduction_indices=[1])
        negative_products_same = tf.reduce_mean(tf.multiply(tiled_negative_feature_same, prediction), reduction_indices=[1])

        for j in xrange(i+1):
            positive_step_scores[j] += [positive_products[:,j,:][:,tf.newaxis,:]]
            negative_step_scores_same[j] += [negative_products_same[:,j,:][:,tf.newaxis,:]]
        
    all_positive_scores, all_negative_scores_same = [], []

    for j in xrange(len(positive_step_scores)):
        all_positive_scores += [tf.concat(positive_step_scores[j], axis=1)]
        all_negative_scores_same += [tf.concat(negative_step_scores_same[j], axis=1)]

    all_positive_scores = tf.concat(all_positive_scores, axis=1)
    all_negative_scores_same = tf.concat(all_negative_scores_same, axis=1)

    return all_positive_scores, all_negative_scores_same

##################################################################################
if VALIDATION: train_set = H5PYDataset(os.path.join(DATA_DIR, '{}_train.hdf5'.format(DATASET)), which_sets=('train',))
else: train_set = H5PYDataset(os.path.join(DATA_DIR, '{}_train.hdf5'.format(DATASET)), which_sets=('train','valid'))

train_stream = DataStream(dataset=train_set, iteration_scheme=SequentialScheme(train_set.num_examples, BATCH_SIZE))
train_iterator = train_stream.get_epoch_iterator()

Ex, ExK, Ex2 = np.zeros((3,)), np.zeros((3,)), np.zeros((3,))
i = 0
K = 0.5*np.ones((1, 3, 256, 256))
while True:
    try: _data = train_iterator.next()
    except StopIteration: break
    _data = leave_out_class(_data, ANOMALY, restore_size=False)
    images = _data[0]/255.

    Ex += np.sum(images, (0,2,3))
    ExK += np.sum(images-K, (0,2,3))
    Ex2 += np.sum((images-K)**2, (0,2,3))
    i += 256*256*images.shape[0]
dataset_mean = Ex/i
dataset_std = np.sqrt((Ex2 - ExK**2/i)/(i-1))

dataset_mean = dataset_mean[None, :, None, None]
dataset_std = dataset_std[None, :, None, None]

##################################################################################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    # Tensor Placeholders:
    all_images = tf.placeholder(tf.int32, shape=[None,3,256,256], name='images')
    all_labels = tf.placeholder(tf.int32, shape=[None], name='labels')

    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')
    selfsup = tf.placeholder(tf.int32, shape=None, name='selfsup')
    lr = tf.placeholder(tf.float32, shape=None, name='lr')

    ####################################################################################
    scaled_images = (tf.cast(all_images, 'float32')/255. - dataset_mean)/dataset_std
    if SELFSUP:
        cpc_images = tf.cond(is_training, lambda: preprocess_images(scaled_images), lambda: center_crop(scaled_images))

        features = extract(cpc_images, selfsup, is_training)
        same_negatives = shuffle(cpc_images, selfsup, is_training)

        context_predictions = context_network(features, cdim=CONTEXT_DIM)
        feature_predictions = predict_features(context_predictions, cdim=CONTEXT_DIM)

        positive_scores, same_negative_scores = compute_scores(features, feature_predictions, same_negatives)

        positive_scores = tf.reshape(positive_scores, [-1, 1])
        same_negative_scores = tf.reshape(same_negative_scores, [-1, 1])

        all_scores = tf.concat([positive_scores, same_negative_scores], -1)
        zero_labels = tf.zeros((tf.shape(positive_scores)[0],), dtype=tf.dtypes.int32)
        nce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=zero_labels, logits=all_scores))
        cpc_loss = WC*nce_loss 

        cpc_optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        cpc_gv = cpc_optimizer.compute_gradients(cpc_loss, var_list=tf.trainable_variables())

    ######################################################################################
    classifier_images = tf.cond(is_training, lambda: preprocess_images(scaled_images), lambda: center_crop(scaled_images))
    classifier_output = cpc_classifier(classifier_images, OUTPUT_DIM-1, selfsup, is_training)

    class_predictions = tf.cast(tf.argmax(classifier_output, axis=1), 'int32')
    class_xentropy_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=all_labels, logits=classifier_output))
    class_accuracy = tf.contrib.metrics.accuracy(labels=all_labels, predictions=class_predictions)

    classifier_train_vars = [param for param in tf.trainable_variables() if 'classifier' in param.name]
    classifier_L2_regularization = tf.add_n([tf.nn.l2_loss(v) for v in classifier_train_vars]) * 5e-4
    class_cost = class_xentropy_cost + classifier_L2_regularization

    classifier_optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    classifier_gv = classifier_optimizer.compute_gradients(class_cost, var_list=classifier_train_vars)

    ######################################################################################
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        class_train_op = classifier_optimizer.apply_gradients(classifier_gv)
        if SELFSUP: cpc_train_op = cpc_optimizer.apply_gradients(cpc_gv)

    ################################### data loaders #####################################
    if VALIDATION: train_set = H5PYDataset(os.path.join(DATA_DIR, '{}_train.hdf5'.format(DATASET)), which_sets=('train',))
    else: train_set = H5PYDataset(os.path.join(DATA_DIR, '{}_train.hdf5'.format(DATASET)), which_sets=('train','valid'))

    train_stream = DataStream(dataset=train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, BATCH_SIZE))
    train_iterator = train_stream.get_epoch_iterator()

    if SELFSUP:
        cpc_stream = DataStream(dataset=train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, CPC_BATCHSIZE))
        cpc_iterator = cpc_stream.get_epoch_iterator()

    if VALIDATION: val_set = H5PYDataset(os.path.join(DATA_DIR, '{}_train.hdf5'.format(DATASET)), which_sets=('valid',))
    else: val_set = H5PYDataset(os.path.join(DATA_DIR, '{}_test.hdf5'.format(DATASET)), which_sets=('test',))

    val_stream = DataStream(dataset=val_set, iteration_scheme=SequentialScheme(val_set.num_examples, BATCH_SIZE))
    ######################################################################################

    ################################## Anomaly detection #################################
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
    odin_classifier_output = cpc_classifier(preprocessed_image, OUTPUT_DIM-1, selfsup, is_training)
    odin_sfx = tf.nn.softmax(odin_classifier_output/ODIN_TEMPERATURE)
    odin_min_dist = tf.reduce_max(odin_sfx, axis=1)

    def signal(_images):
        _msp_signal, _odin_signal = session.run([msp_min_dist, odin_min_dist], 
                feed_dict={all_images: _images, is_training: False, selfsup: 0})
        return -_msp_signal, -_odin_signal

    def compute_av_prec():
        msp_ins, msp_outs, odin_ins, odin_outs = [], [], [], []

        val_iterator = val_stream.get_epoch_iterator()
        while True:
            try:
                _data = val_iterator.next()
                _images, _labels = _data[0], _data[1]
            except StopIteration: break

            in_dist_pos = np.in1d(_labels, np.setdiff1d(range(OUTPUT_DIM), ANOMALY))
            out_dist_pos = np.in1d(_labels, [ANOMALY])

            msp_scores, odin_scores = signal(_images)

            msp_ins += [msp_scores[in_dist_pos]]
            msp_outs += [msp_scores[out_dist_pos]]

            odin_ins += [odin_scores[in_dist_pos]]
            odin_outs += [odin_scores[out_dist_pos]]

        msp_ins, msp_outs = np.concatenate(msp_ins), np.concatenate(msp_outs)
        odin_ins, odin_outs = np.concatenate(odin_ins), np.concatenate(odin_outs)

        TRUE_LABELS = np.hstack((np.zeros(len(msp_ins)), np.ones(len(msp_outs))))
        MSP_TESTS = np.hstack((msp_ins, msp_outs))
        ODIN_TESTS = np.hstack((odin_ins, odin_outs))

        msp_av_prec = 100.*sklearn.metrics.average_precision_score(TRUE_LABELS, MSP_TESTS)
        odin_av_prec = 100.*sklearn.metrics.average_precision_score(TRUE_LABELS, ODIN_TESTS)

        return msp_av_prec, odin_av_prec

    ####################################################################################
    iters_per_epoch = (train_set.num_examples-1300)/BATCH_SIZE
    TOTAL_ITERS = iters_per_epoch*200
    ####################################################################################

    ####################################################################################
    session.run(tf.initialize_all_variables())
    running_avg_cost, running_avg_accuracy = 0.0, 0.0
    tr_costs, dev_costs, tr_accuracies, dev_accuracies = [], [], [], []
    av_precs = []
    
    if SELFSUP:
        running_avg_cpc_cost = 0.0
        cpc_tr_costs = []

    old_epoch = 0
    for iteration in range(TOTAL_ITERS):
        epoch = iteration/iters_per_epoch

        if epoch < 60: _lr = LR
        elif epoch >= 60 and epoch < 120: _lr = LR*0.2
        elif epoch >= 120 and epoch < 160: _lr = LR*0.2*0.2
        elif epoch >= 160: _lr = LR*0.2*0.2*0.2

        if iteration > TOTAL_ITERS: break
        
        ####################################################################
        if SELFSUP:
            try: _data = cpc_iterator.next()
            except StopIteration:
                cpc_stream = DataStream(dataset=train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, CPC_BATCHSIZE))
                cpc_iterator = cpc_stream.get_epoch_iterator()
                _data = cpc_iterator.next()

            # Because Fuel doesn't return full batch-size, hack it:
            if _data[0].shape[0] < CPC_BATCHSIZE:
                cpc_iterator = cpc_stream.get_epoch_iterator()
                _extra_data = cpc_iterator.next()
                _new_data = np.concatenate((_data[0], _extra_data[0]), axis=0)
                _new_labels = np.concatenate((_data[1], _extra_data[1]), axis=0)
                _data = (_new_data, _new_labels)

            _data = leave_out_class(_data, ANOMALY)
            _nce_loss, _ = session.run([nce_loss, cpc_train_op], 
                    feed_dict={all_images: _data[0][:CPC_BATCHSIZE], lr: _lr, is_training: True, selfsup: 1})
            running_avg_cpc_cost += (_nce_loss - running_avg_cpc_cost)/(iteration+1)
            cpc_tr_costs += [running_avg_cpc_cost]

        #####################################################################
        try: _data = train_iterator.next()
        except StopIteration:
            train_stream = DataStream(dataset=train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, BATCH_SIZE))
            train_iterator = train_stream.get_epoch_iterator()
            _data = train_iterator.next()

        if _data[0].shape[0] < BATCH_SIZE:
            train_iterator = train_stream.get_epoch_iterator()
            _extra_data = train_iterator.next()
            _new_data = np.concatenate((_data[0], _extra_data[0]), axis=0)
            _new_labels = np.concatenate((_data[1], _extra_data[1]), axis=0)
            _data = (_new_data, _new_labels)

        _data = leave_out_class(_data, ANOMALY)
        _classifier_cost, _classification_accuracy, _ = session.run([class_xentropy_cost, class_accuracy, class_train_op],
                feed_dict={all_images: _data[0], all_labels: _data[1], lr: _lr, is_training: True, selfsup: 0}) 

        running_avg_cost += (_classifier_cost - running_avg_cost)/(iteration+1)
        running_avg_accuracy += (_classification_accuracy - running_avg_accuracy)/(iteration+1)

        ############################################################
        if iteration % TIMES['print_every'] == 0:
            print 'iter', iteration, '(epoch =', epoch, '): classifier cost =', running_avg_cost, 'accuracy =', running_avg_accuracy,
            if SELFSUP: print 'cpc cost =', running_avg_cpc_cost
            else: print ''

        if iteration % TIMES['test_every'] == 0:
            _dev_cost, _dev_acc = 0.0, 0.0
            _dev_iter = 0
            val_iterator = val_stream.get_epoch_iterator()
            while True:
                try: _data = val_iterator.next()
                except StopIteration:
                    break
                _data = leave_out_class(_data, ANOMALY, restore_size=False)

                _cost, _acc = session.run([class_xentropy_cost, class_accuracy], 
                        feed_dict={all_images: _data[0], all_labels: _data[1], is_training: False, selfsup: 0})

                _dev_cost += (_cost - _dev_cost)/(_dev_iter+1)
                _dev_acc += (_acc - _dev_acc)/(_dev_iter+1)
                _dev_iter += 1

            print ' '*20, '--  dev cost =', _dev_cost, ', dev acc. rate =', _dev_acc, compute_av_prec()

            tr_costs += [running_avg_cost]
            dev_costs += [_dev_cost]
            tr_accuracies += [running_avg_accuracy]
            dev_accuracies += [_dev_acc]

        if epoch != old_epoch and epoch > 90:
            old_epoch = epoch

            av_precs += [compute_av_prec()]

        if iteration == TOTAL_ITERS-1:
            av_precs += [compute_av_prec()]

# Everything keeps wiggling around quite a bit till the end, so average over the last few epochs 
print 'Final: Classification accuracy = {}, Average precision (MSP, ODIN) = {}'.format(np.mean(dev_accuracies[-10:]), np.mean(av_precs[-10:], 0))
