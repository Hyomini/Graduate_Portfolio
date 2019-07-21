import tensorflow as tf
import numpy as np


var = tf.Variable(tf.random_uniform([2, 3]), name="var")
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(var))