import os, sys, time
import numpy as np

import scipy.misc
import read_stl

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
            yield (np.transpose(images[i*batch_size:(i+1)*batch_size], [0, 3, 1, 2]),
                   labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(train_batch_size=50, test_batch_size=50, data_dir=os.environ['DATA_DIR'], remove_class=None, validation=False):

    TRAIN_DATA_PATH = os.path.join(data_dir, 'stl10_binary', 'train_X.bin')
    TRAIN_LABEL_PATH = os.path.join(data_dir, 'stl10_binary', 'train_y.bin')
    TEST_DATA_PATH = os.path.join(data_dir, 'stl10_binary', 'test_X.bin')
    TEST_LABEL_PATH = os.path.join(data_dir, 'stl10_binary', 'test_y.bin')

    train_images = read_stl.read_all_images(TRAIN_DATA_PATH)
    train_labels = read_stl.read_labels(TRAIN_LABEL_PATH)

    if validation:
        rng_state = np.random.get_state()
        np.random.shuffle(train_images)
        np.random.set_state(rng_state)
        np.random.shuffle(train_labels)

        split_point = int(train_images.shape[0]*0.8)
        test_images = train_images[split_point:]
        test_labels = train_labels[split_point:]
        train_images = train_images[:split_point]
        train_labels = train_labels[:split_point]
    else:
        test_images = read_stl.read_all_images(TEST_DATA_PATH)
        test_labels = read_stl.read_labels(TEST_LABEL_PATH)

    if remove_class is not None:
        if not isinstance(remove_class, int): raise Exception('I take exception to this.')
        in_dist_pos = np.in1d(train_labels, np.setdiff1d(range(10), remove_class))
        train_images = train_images[in_dist_pos, ...]
        train_labels = train_labels[in_dist_pos, ...]

        # shift class-ids downward bc you left one out:
        sub = np.where(train_labels > remove_class, 1, 0)
        train_labels = train_labels - sub

        in_dist_pos = np.in1d(test_labels, np.setdiff1d(range(10), remove_class))
        out_dist_pos = np.in1d(test_labels, remove_class)

        in_dist_test_images = test_images[in_dist_pos, ...]
        in_dist_test_labels = test_labels[in_dist_pos, ...]

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
