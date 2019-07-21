from data import input_setup

import time
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import numpy as np
import tensorflow as tf


class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size=32,
                 label_size=32,
                 batch_size=128,
                 c_dim=1,
                 keep_prob=0.5,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim
        self.keep_prob = keep_prob

        #Mahalanobis
        self.mu = 0
        self.sigma = 0.1

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.one_hot = tf.one_hot(self.labels, 10)

        layer_depth = {'L1': 6, 'L2': 16, 'L3': 120, 'L4': 84, 'L5': 10}
        self.weights = {
            'conv1_w': tf.Variable(
            np.sqrt(2.0 / 32 * 32) * tf.truncated_normal(shape=[5, 5, 1, layer_depth.get('L1')],
                                                        mean=self.mu, stddev=self.sigma),name='conv1_w'),
            'conv2_w': tf.Variable(
            np.sqrt(2.0 / 14 * 14) * tf.truncated_normal(shape=[5, 5, layer_depth.get('L1'), layer_depth.get('L2')],
                                                         mean=self.mu, stddev=self.sigma), name='conv2_w'),
            'fullc1_w': tf.Variable(
            np.sqrt(2.0 / 400) * tf.truncated_normal(shape=(400, layer_depth.get('L3')),
                                                    mean=self.mu, stddev=self.sigma),name='fullc1_w'),
            'fullc2_w': tf.Variable(
            np.sqrt(2.0 / 120) * tf.truncated_normal(shape=(layer_depth.get('L3'), layer_depth.get('L4')), mean=self.mu,
                                                     stddev=self.sigma), name='fullc2_w'),
            'fullc3_w': tf.Variable(
            tf.truncated_normal(shape=(layer_depth.get('L4'), layer_depth.get('L5')), mean=self.mu, stddev=self.sigma),
            name='fullc3_w')

        }
        self.biases = {
            'conv1_b': tf.Variable(tf.zeros(layer_depth.get('L1')), name='conv1_b'),
            'conv2_b': tf.Variable(tf.zeros(layer_depth.get('L2')), name='conv2_b'),
            'fullc1_b': tf.Variable(tf.zeros(layer_depth.get('L3')), name='fullc1_b'),
            'fullc2_b': tf.Variable(tf.zeros(layer_depth.get('L4')), name='fullc2_b'),
            'fullc3_b': tf.Variable(tf.zeros(layer_depth.get('L5')), name='fullc3_b')
        }

        self.pred = self.model()

        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.one_hot))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.one_hot, 1)), tf.float32))

        self.saver = tf.train.Saver()

    def train(self, config):

        train_data, valid_data, train_label, valid_label = input_setup(config)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")

            # Tensor Board
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            summary = tf.summary.merge_all()

            writer = tf.summary.FileWriter('./logs')
            writer.add_graph(self.sess.graph)

            for ep in range(config.epoch):
                # Run by batch images
                train_data, train_label = shuffle(train_data, train_label)
                batch_idxs = len(train_data) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]
                    counter += 1
                    _, s, loss, acc = self.sess.run([self.train_op, summary, self.loss, self.accuracy],
                                               feed_dict={self.images: batch_images, self.labels: batch_labels})
                    writer.add_summary(s, global_step=ep)

                    if ep % 10 == 0 and counter % 40 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f], accuracy: [%.2f%%]" \
                              % ((ep + 1), counter, time.time() - start_time, loss, acc*100))

                    if (ep+1) % 1000 == 0:
                        self.save(config.checkpoint_dir, counter)
            valid_acc = self.accuracy.eval(session=self.sess, feed_dict={self.images: valid_data, self.labels: valid_label})
            print(f'Validation set Accuracy: {valid_acc*100:.2f}%')

        else:
            print("Testing...")

    def model(self):
        # Hyperparameters
        mu = 0
        sigma = 0.1
        layer_depth = {'L1': 6, 'L2': 16, 'L3': 120, 'L4': 84, 'L5': 10}

        # L1: Convolutional ~ (32, 32, 1) -> (28, 28, 6)
        conv1 = tf.nn.conv2d(self.images, self.weights['conv1_w'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['conv1_b']
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.dropout(conv1, self.keep_prob)
        # Pooling ~ (28, 28, 6) -> (14, 14, 6)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # L2: Convolutional ~ (14, 14, 6) -> (10, 10, 16)
        conv2 = tf.nn.conv2d(pool1, self.weights['conv2_w'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['conv2_b']
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        # Pooling ~ (10, 10, 16) -> (5, 5, 16)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten ~ (5, 5, 16) -> (400)
        fullc1 = tf.contrib.layers.flatten(pool2)

        # L3: Fully-connected ~ (400) -> (120)
        fullc1 = tf.matmul(fullc1, self.weights['fullc1_w']) + self.biases['fullc1_b']
        fullc1 = tf.nn.relu(fullc1)
        fullc1 = tf.nn.dropout(fullc1, self.keep_prob)

        # L4: Fully-connected ~ (120) -> (84)
        fullc2 = tf.matmul(fullc1, self.weights['fullc2_w']) + self.biases['fullc2_b']
        fullc2 = tf.nn.relu(fullc2)
        fullc2 = tf.nn.dropout(fullc2, self.keep_prob)

        # L5: Fully-connected ~ (84) -> (10)
        # logits(outputs): 입력데이터를 네트워크에서 feed-forward를 통해 나온 결과물
        logits = tf.matmul(fullc2, self.weights['fullc3_w']) + self.biases['fullc3_b']

        return logits

    def save(self, checkpoint_dir, step):
        model_name = "Mnist.model"
        model_dir = "%s_%s" % ("mnist", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("mnist", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
