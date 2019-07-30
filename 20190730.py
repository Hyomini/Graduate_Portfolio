import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pylab as plt
from scipy.spatial import distance
from sklearn.utils import shuffle
import math
from skimage import util
import PIL.Image as pilimg
import warnings


# Ignore WARNINGS
warnings.filterwarnings(action="ignore")

# Set random seed for data permutation and tf.Variable initialization
np.random.seed(0)
tf.set_random_seed(1234)

# Directory(named 'model') for storing trained model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)


# LeNet-5(https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb) ~ 'L3':120->128, 'L4':84->64
def lenet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {'L1': 6, 'L2': 16, 'L3': 128, 'L4': 64, 'L5': 10}

    # Make the variable to global one for generative classifier
    global conv1, conv2, fullc1, fullc2

    # L1: Convolutional ~ (32, 32, 1) -> (28, 28, 6)
    conv1_w = tf.Variable(np.sqrt(2.0/32*32)*tf.truncated_normal(shape=[5, 5, 1, layer_depth.get('L1')], mean=mu, stddev=sigma), name='conv1_w')
    conv1_b = tf.Variable(tf.zeros(layer_depth.get('L1')), name='conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # Pooling ~ (28, 28, 6) -> (14, 14, 6)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # L2: Convolutional ~ (14, 14, 6) -> (10, 10, 16)
    conv2_w = tf.Variable(np.sqrt(2.0/14*14)*tf.truncated_normal(shape=[5, 5, layer_depth.get('L1'), layer_depth.get('L2')], mean=mu, stddev=sigma), name='conv2_w')
    conv2_b = tf.Variable(tf.zeros(layer_depth.get('L2')), name='conv2_b')
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # Pooling ~ (10, 10, 16) -> (5, 5, 16)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten ~ (5, 5, 16) -> (400)
    fullc1 = tf.contrib.layers.flatten(pool2)

    # L3: Fully-connected ~ (400) -> (128)
    fullc1_w = tf.Variable(np.sqrt(2.0/400)*tf.truncated_normal(shape=(400, layer_depth.get('L3')), mean=mu, stddev=sigma), name='fullc1_w')
    fullc1_b = tf.Variable(tf.zeros(layer_depth.get('L3')), name='fullc1_b')
    fullc1 = tf.matmul(fullc1, fullc1_w) + fullc1_b
    fullc1 = tf.nn.relu(fullc1)
    fullc1 = tf.nn.dropout(fullc1, keep_prob)

    # L4: Fully-connected ~ (128) -> (64)
    fullc2_w = tf.Variable(np.sqrt(2.0/120)*tf.truncated_normal(shape=(layer_depth.get('L3'), layer_depth.get('L4')), mean=mu, stddev=sigma), name='fullc2_w')
    fullc2_b = tf.Variable(tf.zeros(layer_depth.get('L4')), name='fullc2_b')
    fullc2 = tf.matmul(fullc1, fullc2_w) + fullc2_b
    fullc2 = tf.nn.relu(fullc2)
    fullc2 = tf.nn.dropout(fullc2, keep_prob)

    # L5: Fully-connected ~ (64) -> (10)
    fullc3_w = tf.Variable(tf.truncated_normal(shape=(layer_depth.get('L4'), layer_depth.get('L5')), mean=mu, stddev=sigma), name='fullc3_w')
    fullc3_b = tf.Variable(tf.zeros(layer_depth.get('L5')), name='fullc3_b')
    # logits(outputs): 입력데이터를 네트워크에서 feed-forward를 통해 나온 결과물
    logits = tf.matmul(fullc2, fullc3_w) + fullc3_b

    return logits


def loss(y, t):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t))


def train_step(loss):
    return tf.train.AdamOptimizer(1e-3).minimize(loss)


def accuracy(y, t):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)), tf.float32))


if __name__ == '__main__':
    # Get data =========================================================================================================
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
    X_test, Y_test = mnist.test.images, mnist.test.labels
    '''
    print("Image Shape: {}".format(X_train[0].shape))        # (28, 28, 1)
    print("Image Shape: {}".format(Y_train[0].shape))                 # ()
    print("Training Set:   {} samples".format(len(X_train)))       # 55000
    print("Validation Set: {} samples".format(len(X_validation)))   # 5000
    print("Test Set:       {} samples".format(len(X_test)))        # 10000
    '''
    # Get random N training data
    n = len(X_train)
    N = 30000
    indices = np.random.permutation(range(n))[:N]
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    # Add padding to training dataset
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')                      # constant(default): 0
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    # Set model ========================================================================================================
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    t = tf.placeholder(tf.int32, shape=[None])
    one_hot_t = tf.one_hot(t, 10)
    keep_prob = tf.placeholder(tf.float32)
    y = lenet(x)
    loss = loss(y, one_hot_t)
    train_step = train_step(loss)
    accuracy = accuracy(y, one_hot_t)

    # Train and evaluate model =========================================================================================
    # 1.Store(Train) ---------------------------------------------------------------------------------------------------
    '''   
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    epochs = 25
    batch_size = 200
    n_batches = N // batch_size
    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end], keep_prob: 0.8})
        val_loss = loss.eval(session=sess, feed_dict={x: X_, t: Y_, keep_prob: 1.0})
        val_acc = accuracy.eval(session=sess, feed_dict={x: X_, t: Y_, keep_prob: 1.0})
        print('epoch:', epoch+1, ' loss:', val_loss, ' accuracy:', val_acc)
    val_acc_v = accuracy.eval(session=sess, feed_dict={x: X_validation, t: Y_validation, keep_prob: 1.0})
    val_acc_t = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test, keep_prob: 1.0})
    print()
    print('validation accuracy: {:.4f}'.format(val_acc_v))
    print('test accuracy:       {:.4f}'.format(val_acc_t))
    print()
    model_path = saver.save(sess, MODEL_DIR + './model.ckpt')
    print('Model saved to:', model_path)
    print()
    '''
    # 2.Restore --------------------------------------------------------------------------------------------------------
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, MODEL_DIR + '/model.ckpt')
    val_acc_v = accuracy.eval(session=sess, feed_dict={x: X_validation, t: Y_validation, keep_prob: 1.0})
    val_acc_t = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test, keep_prob: 1.0})
    print()
    print('validation accuracy: {:.4f}'.format(val_acc_v))
    print('test accuracy:       {:.4f}'.format(val_acc_t))
    print()
    # (epochs, batch_size, keep_drop) = (25, 200, 0.8) 일 때, (validation accuracy, test accuracy) = (0.9892, 0.9891)



########################################################################################################################
# Generative classifier ================================================================================================
N = len(X_train)                                                                        # number of training data: 30000
label_of_x = np.array(sess.run(tf.argmax(y, 1), feed_dict={x: X_train, keep_prob: 1.0}))    # classified label of data x
f_x = np.array(sess.run(fullc2, feed_dict={x: X_train, keep_prob: 1.0}))   # hidden layer's output of data x[i]: f(x[i])
num_of_labels = 10                                                                         # number of labels: 10(Mnist)
num_of_neurons = len(f_x[0])                                     # number of neurons(dimensions) in the hidden layer: 64

# Data(data[i][j]: (j+1)th data of label i): get f(x) and distinguish it by its label ----------------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()                                                          # make temporary list to use appending
for i in range(N):
    temp[label_of_x[i]].append(f_x[i])
data = np.array(temp)                                                                       # change list to numpy array

# Mu_hat(mean vector for each label, mu_hat[i] = mean vector of label i): get data and calculate mean in each label ----
mu_hat = [np.zeros(num_of_neurons)] * num_of_labels
for label in range(num_of_labels):
    mu_hat[label] = np.mean(data[label], axis=0)

# Sigma_hat(tied covariance of class-conditional distribution of f(x[i]): outer-product to all data and calculate mean -
sigma_hat = np.zeros((num_of_neurons, num_of_neurons))
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (num_of_neurons, 1))
        v = np.reshape(mu_hat[label], (num_of_neurons, 1))
        sigma_hat += np.matmul((u - v), (u - v).T)
sigma_hat = sigma_hat / N
'''
print(np.linalg.det(sigma_hat))
'''

# Calculate Mahalanobis distance of f(x[i]) in each label's distribution -----------------------------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()                                                          # make temporary list to use appending
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        # temp[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
# euclidean_dist_data = np.array(temp)
mahalanobis_dist_data = np.array(temp)                                                      # change list to numpy array

# Set distance threshold for detecting OOD in each label ---------------------------------------------------------------
temp = [None]*num_of_labels
for label in range(num_of_labels):
    temp[label] = list()                                                          # make temporary list to use appending
for label in range(num_of_labels):
    # temp[label].append(np.percentile(euclidean_dist_data[label], 95, interpolation='linear'))
    temp[label].append(np.percentile(mahalanobis_dist_data[label], 95, interpolation='linear'))
# euclidean_dist_threshold = np.array(temp)
mahalanobis_dist_threshold = np.array(temp)                                                 # change list to numpy array
print("<OOD threshold of distance of penultimate logits in each label>")
print(mahalanobis_dist_threshold, end='\n')

# Show histogram for each label's distance distribution ----------------------------------------------------------------
'''
for label in range(10):
    data = np.sort(mahalanobis_dist_data[label])
    bins = np.arange(0, 300, 2)
    plt.hist(data, bins, normed=True)
    plt.title("label: %d" % label)
    plt.xlabel('distance', fontsize=15)
    plt.ylabel('num of data', fontsize=15)
    plt.show(block=True)
'''

# Show grey image of a random test data and classify it ----------------------------------------------------------------
'''
X_test_f_x = sess.run(fullc2, feed_dict={x: X_test, keep_prob: 1.0})
random_idx = 0
random_data = X_test[random_idx]
random_data_f_x = X_test_f_x[random_idx]
plt.figure(figsize=(5, 5))
plt.imshow(np.reshape(random_data, [32, 32]), cmap='Greys')
plt.show()
temp = [None]*num_labels
for label in range(num_labels):
    temp[label] = list()
    u = np.reshape(random_data_f_x, (1, num_neurons))
    v = np.reshape(mu_hat[label], (1, num_neurons))
    temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat))**2)
dist_to_each_label_mean = np.array(temp)
print('<Mahalanobis distance of the image from each label\'s mean in penultimate logits>')
print(dist_to_each_label_mean)
print()
idx = np.argmin(dist_to_each_label_mean, 0)                                               # find the nearest label index
if dist_to_each_label_mean[idx] >= m_dist_threshold[idx]:                                       # and check if it is OOD
    print('The image is classified to \'OOD\' by Mahalanobis distance classifier', end='\n')
else:
    print('The image is classified to \'' + str(int(idx)) + '\' by Mahalanobis distance classifier', end='\n')
'''


# Test the generative classifier using Mahalanobis distance on test dataset ============================================
N = len(X_test)
f_x_of_X_test = sess.run(fullc2, feed_dict={x: X_test, keep_prob: 1.0})
label_of_X_test = np.array(range(N))
target_label_of_X_test = Y_test
for i in range(N):
    temp = [None] * num_of_labels
    for label in range(num_of_labels):
        temp[label] = list()
        u = np.reshape(f_x_of_X_test[i], (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat))**2)
    mahalanobis_dist_data_of_X_test = np.array(temp)
    idx = np.argmin(mahalanobis_dist_data_of_X_test, 0)                                 # find the nearest label's index
    if mahalanobis_dist_data_of_X_test[idx] < mahalanobis_dist_threshold[idx]:               # if it is in-distribution,
        label_of_X_test[i] = idx                                                               # then give label's index
    else:                                                                           # if it is OOD(out-of-distribution),
        label_of_X_test[i] = idx + 10                                        # then give OOD index(= label's index + 10)

# Print the accuracy of in-distribution data ---------------------------------------------------------------------------
num_of_in_distribution = 0
num_of_correctly_classified = 0
in_distribution_accuracy = 0.0
for i in range(N):
    if label_of_X_test[i] < 10:
        num_of_in_distribution = num_of_in_distribution + 1
        if label_of_X_test[i] == target_label_of_X_test[i]:
            num_of_correctly_classified = num_of_correctly_classified + 1
in_distribution_accuracy = num_of_correctly_classified / num_of_in_distribution
print(in_distribution_accuracy, end='\n')

# Show image of the label's OOD data and its distance ------------------------------------------------------------------
chosen_label = 1                                                                                                 # 0 ~ 9
for i in range(N):
    if label_of_X_test[i] == chosen_label + 10:
        print("target: %f label: %f mahalanobis_dist_data: %f"
              % (target_label_of_X_test, label_of_X_test[i], mahalanobis_dist_data[i]))
        # plt.title("Mahalanobis distance to the label(%f)'s mean: %f" % (label_of_X_test[i], mahalanobis_dist_data[i]))
        plt.figure(figsize=(5, 5))
        plt.imshow(np.reshape(X_test[i], [32, 32]), cmap='Greys')
        plt.show()
