import os, sys
sys.path.append(os.getcwd())
sys.path.append('../')

import dataloaders.cifar10

import numpy as np
import scipy.stats
import sklearn.metrics

from scipy import ndimage

import time
import random
import functools
import cPickle as pkl

IMDIM = 32*32

def dequantize(x):
    return x + (1./256)*np.random.uniform(size=x.shape)  

BS = 5000
train_data, dev_data  = dataloaders.cifar10.load(train_batch_size=BS,
                                                 test_batch_size=BS,
                                                 data_dir=os.environ['DATA_DIR'])
all_train_data = []
train_gen = train_data()
for i in range(50000/BS):
    _data = train_gen.next()
    all_train_data += [dequantize(_data[0]/256.)]
all_train_data = np.concatenate(all_train_data)
all_train_data = np.reshape(all_train_data, [-1, 3, IMDIM]).astype('float32')

MEAN1 = np.mean(all_train_data[:, 0, ...], axis=0).astype('float32')
COV1 = np.cov(all_train_data[:, 0, ...], rowvar=False).astype('float32')

MEAN2 = np.mean(all_train_data[:, 1, ...], axis=0).astype('float32')
COV2 = np.cov(all_train_data[:, 1, ...], rowvar=False).astype('float32')

MEAN3 = np.mean(all_train_data[:, 2, ...], axis=0).astype('float32')
COV3 = np.cov(all_train_data[:, 2, ...], rowvar=False).astype('float32')

def nllhood(images):
    pdfs1 = scipy.stats.multivariate_normal.logpdf(images[:, 0, ...], mean=MEAN1, cov=COV1)
    pdfs2 = scipy.stats.multivariate_normal.logpdf(images[:, 1, ...], mean=MEAN2, cov=COV2)
    pdfs3 = scipy.stats.multivariate_normal.logpdf(images[:, 2, ...], mean=MEAN3, cov=COV3)
    pdfs = pdfs1 + pdfs2 + pdfs3
    return -pdfs

in_data = []
dev_gen = dev_data()
for i in range(10000/BS):
    _data = dev_gen.next()
    in_data += [dequantize(_data[0]/256.)]
all_in_data = np.concatenate(in_data)
all_in_data = np.reshape(all_in_data, [-1, 3, IMDIM]).astype('float32')

datasets = ['Imagenet', 'Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN']

in_test_signal = np.asarray(nllhood(all_in_data))
for dataset in datasets:
    # The pickle files are numpy arrays of the images from https://github.com/facebookresearch/odin
    D = pkl.load(open('/scratch/faruk/data/OOD/{}.pkl'.format(dataset), 'rb'))

    all_data = []
    for i in range(10000/BS):
        _data = D[i*BS:(i+1)*BS]
        all_data += [dequantize(_data/256.)]
    all_out_data = np.concatenate(all_data)
    all_out_data = np.reshape(all_out_data, [-1, 3, IMDIM]).astype('float32')

    out_test_signal = np.asarray(nllhood(all_out_data))

    TRUE_LABELS = np.hstack((np.zeros(len(in_test_signal)), np.ones(len(out_test_signal))))
    TEST_LLS = np.hstack((in_test_signal, out_test_signal))

    aupr = 100.*sklearn.metrics.average_precision_score(TRUE_LABELS, TEST_LLS)
    print dataset, '\t', aupr
