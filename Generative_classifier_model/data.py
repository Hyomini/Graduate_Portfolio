import os
import glob
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def input_setup(config):
    # Load data path
    mnist = input_data.read_data_sets(config.sample_dir, reshape=False)
    x_valid, y_valid = mnist.validation.images, mnist.validation.labels
    if config.is_train:
        x_, y_ = mnist.train.images, mnist.train.labels
        length_data = len(x_)
        N = 30000
        indices = np.random.permutation(range(length_data))[:N]
        x_ = x_[indices]
        y_ = y_[indices]
    else:
        x_, y_ = mnist.test.images, mnist.test.labels

    # Add padding to training dataset
    x_ = np.pad(x_, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # constant_values's default is 0
    x_valid = np.pad(x_valid, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    return x_, x_valid, y_, y_valid