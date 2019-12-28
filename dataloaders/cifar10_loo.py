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

def load_data(filenames, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    images = images.reshape((images.shape[0], 3, 32, 32))

    return images, labels

def data_generator(images, labels, batch_size, shuffle=True):

    if images.shape[0]%batch_size != 0:
        how_many_more = batch_size - images.shape[0]%batch_size
        images = np.concatenate((images, images[:how_many_more, ...]), axis=0)
        labels = np.concatenate((labels, labels[:how_many_more, ...]), axis=0)

    def get_epoch():
        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch

def load(train_batch_size, test_batch_size, data_dir=os.environ['DATA_DIR'], remove_class=None, validation=False):
    dirname = data_dir + 'cifar-10-batches-py'

    if validation:
        train_images, train_labels = load_data(['data_batch_1','data_batch_2','data_batch_3','data_batch_4'], dirname)
        test_images, test_labels = load_data(['data_batch_5'], dirname)
    else:
        train_images, train_labels = load_data(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], dirname)
        test_images, test_labels = load_data(['test_batch'], dirname)

    if remove_class is not None:
        if not isinstance(remove_class, int): raise Exception('I take exception to this.')
        in_dist_pos = np.in1d(train_labels, np.setdiff1d(range(10), remove_class))
        train_images = train_images[in_dist_pos, ...]
        train_labels = train_labels[in_dist_pos, ...]

        # shift class-ids downward bc you left one out:
        sub = np.where(train_labels > remove_class, 1, 0)
        train_labels = train_labels - sub

    if remove_class is not None:
        in_dist_pos = np.in1d(test_labels, np.setdiff1d(range(10), remove_class))
        out_dist_pos = np.in1d(test_labels, remove_class)

        in_dist_test_images = test_images[in_dist_pos, ...]
        in_dist_test_labels = test_labels[in_dist_pos, ...]

        # shift class-ids downward bc you left one out:
        sub = np.where(in_dist_test_labels > remove_class, 1, 0)
        in_dist_test_labels = in_dist_test_labels - sub

        out_dist_test_images = test_images[out_dist_pos, ...]
        out_dist_test_labels = test_labels[out_dist_pos, ...]

    if remove_class is not None:
        return (
            data_generator(train_images, train_labels, train_batch_size),
            data_generator(in_dist_test_images, in_dist_test_labels, test_batch_size, shuffle=False),
            data_generator(out_dist_test_images, out_dist_test_labels, test_batch_size, shuffle=False)
        )
    else:
        return (
            data_generator(train_images, train_labels, train_batch_size),
            data_generator(test_images, test_labels, test_batch_size, shuffle=False)
        )
