from data import input_setup

import time
import os
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf


class SRCNN(object):

  def __init__(self,
               sess, 
               image_size=32,
               label_size=32,
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()
  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    
    self.weights = {
      'w1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=1e-3), name='w1'),
      'w2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-3), name='w2'),
      'w3': tf.Variable(tf.random_normal([3, 3, 64, 1], stddev=1e-3), name='w3')
    }
    self.biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([64]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    # PSNR
    self.PSNR = tf.reduce_mean(tf.image.psnr(self.labels, self.pred, 1))

    self.saver = tf.train.Saver()

  def train(self, config):

    train_data, train_label = input_setup(config)

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
      tf.summary.scalar("psnr", self.PSNR)
      summary = tf.summary.merge_all()

      writer = tf.summary.FileWriter('./ logs')
      writer.add_graph(self.sess.graph)

      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        for idx in range(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
          counter += 1
          _, s, loss = self.sess.run([self.train_op, summary, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
          writer.add_summary(s, global_step=ep)

          if ep % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
            % ((ep+1), counter, time.time()-start_time, loss))

          if counter % 1000 == 0:
            self.save(config.checkpoint_dir, counter)

    else:
      print("Testing...")

      result_psnr = self.PSNR.eval({self.images: train_data, self.labels: train_label})
      result = self.pred.eval({self.images: train_data, self.labels: train_label})
      result = Image.fromarray((result[3]*255).squeeze())
      result.show()
      print('Result PSNR: {}'.format(result_psnr))


  def model(self):
    conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
    conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
