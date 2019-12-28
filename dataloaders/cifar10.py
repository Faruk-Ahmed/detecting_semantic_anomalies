import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir, do_not_shuffle=False):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    images = images.reshape((images.shape[0], 3, 32, 32))
    labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        if not do_not_shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(train_batch_size, test_batch_size, data_dir=os.environ['DATA_DIR'], validation=False):
    dirname = data_dir + 'cifar-10-batches-py'

    if not validation:
        return (
            cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], train_batch_size, dirname),
            cifar_generator(['test_batch'], test_batch_size, dirname, do_not_shuffle=True)
        )
    else:
        return (
            cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4'], train_batch_size, dirname),
            cifar_generator(['data_batch_5'], train_batch_size, dirname, do_not_shuffle=True),
        )

